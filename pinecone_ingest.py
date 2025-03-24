import os
import fitz  
import ollama
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec

# Pinecone setup
PINECONE_API_KEY = "pcsk_62H5cT_7YHFMU5JH1PhPRGjjdVRChy8QTfYNdxWj6JsSqAJgbSmCj2hZZc9pkH12h4aUUf" 
INDEX_NAME = "embedding-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    
    # Wait for index initialization
    import time
    time.sleep(20)  

# Connect to index
index = pc.Index(INDEX_NAME)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get embedding for text using Ollama"""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files"""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num + 1, page.get_text()))  
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """Store embedding in Pinecone"""
    vector_id = str(uuid4())  # Unique ID for each chunk
    metadata = {"file": file, "page": page, "text": chunk[:1000]}  # Limit metadata size
    index.upsert(vectors=[(vector_id, embedding, metadata)])
    print(f"Stored embedding for: {chunk[:50]}...")

def process_pdfs(data_dir):
    """Process all PDFs in the data directory"""
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

if __name__ == "__main__":
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    print("Starting PDF processing...")
    process_pdfs(data_dir)
    print("PDF processing complete!")
