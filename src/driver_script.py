import redis
import chromadb
from sentence_transformers import SentenceTransformer
import ingest
import search
import time
import tracemalloc
import os
import uuid
import pandas as pd
import platform

# Embedding Models
embedding_models = {
    "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
    "MPNet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "InstructorXL": SentenceTransformer("hkunlp/instructor-xl"),
}

# Vector DBs
vector_dbs = {"redis": redis.Redis(host="localhost", port=6379, db=0) , "chroma": chromadb.PersistentClient(path="./chroma_db")}

chunk_sizes = [200, 500, 1000]
chunk_overlaps = [0, 50, 100]

test_queries = ["What is an AVL tree?", "When are Professor Fontenot's Office Hours", "What is a data structure?", "Tell me more about hash maps."]
csv_path = "./test_data.csv" 


def measure_time_and_memory(test_query):
    tracemalloc.start() 
    start_time = time.perf_counter()

    search.search_embeddings(test_query)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop() 

    execution_time = end_time - start_time
    memory_used = peak / (1024 * 1024) 

    return execution_time, memory_used


def create_test_results():
    results = []

    for model_name, model in embedding_models.items():
        for db_name, db_client in vector_dbs.items():
            for size in chunk_sizes:
                for query in test_queries:
                    for overlap in chunk_overlaps: 
                        query_time, query_memory = measure_time_and_memory(query)
                        results.append([model_name, db_name, size, overlap, query_time, query_memory])

    df = pd.DataFrame(results, columns=[
        'embedding_model', 'vector_db', 'chunk_size', 'chunk_overlap',
        'query_time', 'query_memory' 
    ])

    df.to_csv(csv_path, index=False)


create_test_results()
