## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
from sentence_transformers import SentenceTransformer

# initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Configuring vector dimensions, index_name, doc_prefix, and distance metric
VECTOR_DIM = 784
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def clear_redis_store():
    '''
    Used to clear the redis vector store
    '''
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index():
    '''
    Create an HNSW index in Redis
    '''
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
    '''
    Takes in text, embedding model, and use_llama boolean, 
    Generate an embedding based on either a llama embedding model or SentenceTransformer
    '''
    if use_llama:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        sentence_transformer = SentenceTransformer(model)
        return sentence_transformer.encode(text).tolist()


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    '''
    Store the embedding in Redis
    '''
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  
        },
    )
    print(f"Stored embedding for: {chunk}")

def extract_text_from_pdf(pdf_path):
    '''
    extract text from a PDF file
    '''
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page



def split_text_into_chunks(text, chunk_size=300, overlap=50):
    '''
    Split text into chunks of approximately chunk_size words with overlap
    '''
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir, model, use_llama=False, chunk_size=300, overlap=50):
    '''
    Process all PDF files in a given directory
    '''
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk, model, use_llama)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_redis(query_text: str, model, use_llama):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text, model, use_llama)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def run_ingest(chunk_size, chunk_overlap, model, use_llama=False):
    '''
    Takes in chunk_size, chunk_overlap, embedding_model, and use_llama boolean, 
    Runs all relevant functions in ingest file for testing purposes
    '''
    clear_redis_store()
    create_hnsw_index()
    process_pdfs("../data/", model, use_llama, chunk_size, chunk_overlap)
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?", "nomic-embed-text", True)

            
            

def main():
    clear_redis_store()
    create_hnsw_index()

    process_pdfs("../data/", "nomic-embed-text", True)
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?", "nomic-embed-text", True)



if __name__ == "__main__":
    main()


