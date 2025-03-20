import redis
import chromadb
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ingest
import search
import time
import tracemalloc
import os
import uuid
import pandas as pd
import platform
from tqdm import tqdm

# Embedding Models
embedding_models = {
    "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
    "MPNet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "InstructorXL": SentenceTransformer("hkunlp/instructor-xl"),
}

llms = {"Mistral": "mistral:latest", "Llama": "llama2:7b"}



# Vector DBs
vector_dbs = {"qdrant": QdrantClient(path="./qdrant_db"), "redis": redis.Redis(host="localhost", port=6379, db=0) , "chroma": chromadb.PersistentClient(path="./chroma_db")}

chunk_sizes = [200, 500, 1000]
chunk_overlaps = [0, 50, 100]

test_queries = ["What is an AVL tree?", "When are Professor Fontenot's Office Hours", "How do you create a hash map?"]
csv_path = "./test_data.csv" 


def measure_time_and_memory(test_query, llm):
    tracemalloc.start() 
    start_time = time.perf_counter()

    context_results = search.search_embeddings(test_query)

    response = search.generate_rag_response(test_query, llm, context_results)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop() 

    execution_time = end_time - start_time
    memory_used = peak / (1024 * 1024) 

    return execution_time, memory_used, response


def create_test_results():
    results = []
    total_combinations = len(llms) * len(embedding_models) * len(vector_dbs) * len(chunk_sizes) * len(test_queries) * len(chunk_overlaps)
    with tqdm(total=total_combinations, desc="Processing Combos") as pbar:
        for llm_name, llm in llms.items(): 
            for model_name, model in embedding_models.items():
                for db_name, db_client in vector_dbs.items():
                    for size in chunk_sizes:
                        for query in test_queries:
                            for overlap in chunk_overlaps: 
                                query_time, query_memory, response = measure_time_and_memory(query, llm)
                                results.append([model_name, db_name, size, overlap, query_time, query_memory, response, llm_name])
                                pbar.update(1)
    df = pd.DataFrame(results, columns=[
        'embedding_model', 'vector_db', 'chunk_size', 'chunk_overlap',
        'query_time', 'query_memory', 'response', 'llm'
    ])

    df.to_csv(csv_path, index=False)


create_test_results()




