import os
import fitz  
import ollama
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec

# initialize Pinecone
PINECONE_API_KEY = "pcsk_62H5cT_7YHFMU5JH1PhPRGjjdVRChy8QTfYNdxWj6JsSqAJgbSmCj2hZZc9pkH12h4aUUf" 
INDEX_NAME = "embedding-index"

# initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    
    # wait for index initialization
    import time
    time.sleep(20)  

# connect to index
index = pc.Index(INDEX_NAME)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """get embedding for text using ollama"""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def extract_text_from_pdf(pdf_path):
    """extract text from PDF files"""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num + 1, page.get_text()))  
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """split text into chunks of approximately chunk_size words with overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """store embedding in Pinecone"""
    vector_id = str(uuid4())  # Unique ID for each chunk
    metadata = {"file": file, "page": page, "text": chunk[:1000]}  # Limit metadata size
    index.upsert(vectors=[(vector_id, embedding, metadata)])
    print(f"Stored embedding for: {chunk[:50]}...")

def process_pdfs(data_dir):
    """process all PDFs in the data directory"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")
        
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            try:
                text_by_page = extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = split_text_into_chunks(text)
                    for chunk in chunks:
                        embedding = get_embedding(chunk)
                        store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=chunk,
                            embedding=embedding
                        )
                print(f"Successfully processed {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

def search_embeddings(query, top_k=3):
    """search Pinecone for the most relevant embeddings"""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

def generate_rag_response(query):
    """generate response using retrieved context"""
    context_results = search_embeddings(query)
    if not context_results:
        return "No relevant context found for your query."
        
    context_str = "\n".join(
        [
            f"From {result.metadata['file']} (page {result.metadata['page']}): {result.metadata['text']}..."
            for result in context_results
        ]
    )
    
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
    
Context:
{context_str}

Question: {query}

Provide a detailed answer based on the context:"""
    
    response = ollama.chat(
        model="llama3.2:latest",  # Updated model name
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def interactive_search():
    """interactive RAG search"""
    print("Pinecone RAG Search Interface")
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
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting PDF processing...")
    process_pdfs(data_dir)
    
    print("\nStarting interactive search...")
    interactive_search()
