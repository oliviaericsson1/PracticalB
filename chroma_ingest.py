import os
import fitz 
import numpy as np
import chromadb
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import ollama
import uuid

# Connect to ChromaDB running in Docker
chroma_client = HttpClient(host="localhost", port=8000)
COLLECTION_NAME = "embedding_collection"
VECTOR_DIM = 768  

collection = None

def clear_chroma_store():
    global collection
    print("Clearing ChromaDB store...")
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        print("Collection does not exist yet.")
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    print("ChromaDB store ready.\n")


def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
    if use_llama:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        sentence_transformer = SentenceTransformer(model)
        return sentence_transformer.encode(text).tolist()


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs(data_dir, model, use_llama=False, chunk_size=300, overlap=50):
    global collection

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model, use_llama)
                    unique_id = str(uuid.uuid4())
                    metadata = {
                        "file": file_name,
                        "page": str(page_num),
                        "chunk_index": str(chunk_index)
                    }
                    collection.add(
                        ids=[unique_id],
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                    print(f"Stored: {file_name} - page {page_num} - chunk {chunk_index}")
            print(f" -----> Done processing {file_name}\n")


def query_chroma(query_text: str, model: str, use_llama: bool = False, top_k=5):
    global collection
    embedding = get_embedding(query_text, model, use_llama)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    print(f"\n Top {top_k} results for: '{query_text}'\n")
    for doc, score, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        print(f"[{meta['file']} | Page {meta['page']} | Chunk {meta['chunk_index']}]")
        print(f"Distance: {score:.4f}")
        print(f"Text: {doc[:200]}...\n")


def run_ingest(chunk_size=300, chunk_overlap=50, model="nomic-embed-text", use_llama=True):
    clear_chroma_store()
    process_pdfs("../data/", model, use_llama, chunk_size, chunk_overlap)
    print("\n Done ingesting all PDFs.\n")
    query_chroma("What is the capital of France?", model, use_llama)


def main():
    run_ingest()


if __name__ == "__main__":
    main()
