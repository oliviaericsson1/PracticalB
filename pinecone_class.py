import os
import fitz  
import ollama
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
import time
from typing import Optional, Union

class EmbeddingManager:
    def __init__(self):
        self.local_models = {}
        self.current_model = None
        self.current_dimension = None
        self.use_llama = False

    def get_model_dimension(self, model_name: str, use_llama: bool = False) -> int:
        """Get the output dimension for a given model"""
        if use_llama:
            # For Ollama models, we need to test with a sample to get dimension
            test_text = "test"
            embedding = ollama.embeddings(model=model_name, prompt=test_text)["embedding"]
            return len(embedding)
        else:
            # For Sentence Transformer models, we can get dimension from the model config
            if model_name not in self.local_models:
                self.local_models[model_name] = SentenceTransformer(model_name)
            return self.local_models[model_name].get_sentence_embedding_dimension()

    def initialize_embedding_model(self, model_name: str, use_llama: bool = False):
        """Initialize the embedding model"""
        self.current_model = model_name
        self.use_llama = use_llama
        self.current_dimension = self.get_model_dimension(model_name, use_llama)
        
        if not use_llama and model_name not in self.local_models:
            self.local_models[model_name] = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding using the current model"""
        if self.use_llama:
            response = ollama.embeddings(model=self.current_model, prompt=text)
            return response["embedding"]
        else:
            return self.local_models[self.current_model].encode(text).tolist()

# Configuration
PINECONE_API_KEY = "pcsk_62H5cT_7YHFMU5JH1PhPRGjjdVRChy8QTfYNdxWj6JsSqAJgbSmCj2hZZc9pkH12h4aUUf" 
INDEX_NAME = "embedding-index"
DATA_DIR = "../data"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_manager = EmbeddingManager()

def setup_pinecone_index(model_name: str, use_llama: bool = False) -> None:
    """Ensure Pinecone index exists with correct dimension"""
    # First get the model dimension
    embedding_manager.initialize_embedding_model(model_name, use_llama)
    required_dimension = embedding_manager.current_dimension

    # Check if existing index needs to be replaced
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME in existing_indexes:
        index_info = pc.describe_index(INDEX_NAME)
        if index_info.dimension != required_dimension:
            pc.delete_index(INDEX_NAME)
            print(f"Deleted existing index with dimension {index_info.dimension}")
            existing_indexes.remove(INDEX_NAME)

    # Create new index if needed
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=required_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new index with dimension {required_dimension}")
        time.sleep(20)  # Wait for index initialization

    # Connect to index
    global index
    index = pc.Index(INDEX_NAME)

def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from PDF files"""
    try:
        doc = fitz.open(pdf_path)
        return [(i+1, page.get_text()) for i, page in enumerate(doc)]
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return []

def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into chunks with overlap"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def store_embedding(file: str, page: str, chunk: str) -> None:
    """Store embedding in Pinecone"""
    try:
        embedding = embedding_manager.get_embedding(chunk)
        vector_id = str(uuid4())
        metadata = {
            "file": file,
            "page": page,
            "text": chunk[:1000],
            "model": embedding_manager.current_model
        }
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        print(f"Stored embedding ({len(embedding)}D) for: {chunk[:50]}...")
    except Exception as e:
        print(f"Error storing embedding: {e}")

def process_pdfs(data_dir: str = DATA_DIR, 
                model_name: str = "all-MiniLM-L6-v2",
                use_llama: bool = False,
                chunk_size: int = 300,
                overlap: int = 50) -> None:
    """Process all PDFs in directory with specified model"""
    # Setup index with correct dimension
    setup_pinecone_index(model_name, use_llama)
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(data_dir, file_name)
        print(f"\nProcessing {file_name} with {model_name}...")
        
        for page_num, text in extract_text_from_pdf(pdf_path):
            for chunk in split_text_into_chunks(text, chunk_size, overlap):
                store_embedding(file_name, str(page_num), chunk)
        
        print(f"Completed processing {file_name}")

def search_embeddings(query: str, top_k: int = 3) -> list:
    """Search Pinecone for similar embeddings"""
    try:
        query_embedding = embedding_manager.get_embedding(query)
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        ).matches
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query: str) -> str:
    """Generate response using retrieved context"""
    context_results = search_embeddings(query)
    if not context_results:
        return "No relevant context found for your query."
        
    context_str = "\n".join(
        f"From {r.metadata['file']} (page {r.metadata['page']}): {r.metadata['text']}..."
        for r in context_results
    )
    
    prompt = f"""Use this context to answer the question:
    
Context:
{context_str}

Question: {query}

Answer:"""
    
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating response: {e}"

def interactive_search() -> None:
    """Interactive RAG search interface"""
    print("\nPinecone RAG Search Interface")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
                
            print("\nProcessing your query...")
            response = generate_rag_response(query)
            print("\n--- Response ---\n")
            print(response)
            print("\n" + "-"*40)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Example usage with different models:
    # process_pdfs(model_name="all-MiniLM-L6-v2")  # 384-dim
    #process_pdfs(model_name="sentence-transformers/all-mpnet-base-v2")  # 768-dim
    process_pdfs(model_name="nomic-embed-text", use_llama=True)  # Ollama model
    
    
    print("\nStarting interactive search...")
    interactive_search()
