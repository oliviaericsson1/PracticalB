import os
import time
import fitz
import uuid
import ollama
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Configuring API_key, INDEX_NAME, Vector dimensions, and data directory
PINECONE_API_KEY = "...enter API key..."
INDEX_NAME = "embedding-index"
VECTOR_DIM = 768 # Modify to 384 for MiniLM embedding model
DATA_DIR = "../data"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize index
index = None


def clear_pinecone_index():
    '''
    Clears Pinecone index before ingestion
    '''
    global index
    print("Resetting Pinecone index...")
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME in existing_indexes:
        pc.delete_index(INDEX_NAME)
        print("Deleted existing index.")

    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(20)
    index = pc.Index(INDEX_NAME)
    print("Pinecone index ready.\n")


def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
     '''
    Takes in text to embed str, embedding model str, and a use_llama boolean,
    Returns the embedding using either a llama or Sentence transformer embedding model
    '''
    if use_llama:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        return SentenceTransformer(model).encode(text).tolist()


def extract_text_from_pdf(pdf_path):
    '''
    Takes in a pdf_path and extracts the text from the pdf
    '''
    doc = fitz.open(pdf_path)
    return [(i, page.get_text()) for i, page in enumerate(doc)]


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    '''
    Takes in the text, chunk size, and overlap,
    Returns the splitted text depending on these inputted chunk and overlap values
    '''
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


def process_pdfs(data_dir, model, use_llama=False, chunk_size=300, overlap=50):
    '''
    Takes in a data directory, embedding model, use_llama boolean, chunk size, and overlap,
    Processes each pdf
    '''
    global index

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".pdf"):
            continue

        print(f"Processing {file_name}...")
        pdf_path = os.path.join(data_dir, file_name)
        pages = extract_text_from_pdf(pdf_path)

        for page_num, text in pages:
            chunks = split_text_into_chunks(text, chunk_size, overlap)
            for chunk_index, chunk in enumerate(chunks):
                embedding = get_embedding(chunk, model, use_llama)
                vector_id = str(uuid.uuid4())
                metadata = {
                    "file": file_name,
                    "page": str(page_num),
                    "chunk_index": str(chunk_index),
                    "model": model
                }
                try:
                    index.upsert(vectors=[(vector_id, embedding, metadata)])
                    print(f"Stored: {file_name} - page {page_num} - chunk {chunk_index}")
                except Exception as e:
                    print(f"Error storing vector: {e}")
        print(f"-----> Done processing {file_name}\n")


def run_ingest(chunk_size=300, chunk_overlap=50, model="nomic-embed-text", use_llama=True):
    '''
    Takes in chunk_size, chunk_overlap, embedding model, and use_llama boolean, and runs all necessary
    functions in this ingest file for testing purposes
    '''
    clear_pinecone_index()
    process_pdfs(DATA_DIR, model, use_llama, chunk_size, chunk_overlap)
    print("\n Done ingesting all PDFs.\n")


def main():
    run_ingest()


if __name__ == "__main__":
    main()
