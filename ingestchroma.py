import ollama
import chromadb
import numpy as np
import os
import fitz

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection = chroma_client.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768

# Clear ChromaDB collection
def clear_chromadb_store():
    print("Clearing existing ChromaDB store...")
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        print("ChromaDB store cleared.")
    else:
        print("No documents found in ChromaDB.")

# Generate an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store embedding in ChromaDB
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    print(f"Stored embedding for: {chunk[:30]}...")

# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Process all PDF files in a given directory
def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

# Query ChromaDB for similar embeddings
def query_chromadb(query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    
    for doc in results["ids"][0]:
        print(f"Document ID: {doc}")
    
    print("\nQuery results:")
    for i, metadata in enumerate(results["metadatas"][0]):
        print(f"{i+1}. File: {metadata['file']}, Page: {metadata['page']}, Chunk: {metadata['chunk'][:30]}...")

# Main function
def main():
    clear_chromadb_store()
    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_chromadb("What is the capital of France?")

if __name__ == "__main__":
    main()