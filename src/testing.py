import redis_ingest
import redis_search
import chroma_ingest
import chroma_search
import time
import tracemalloc
import os
import pandas as pd
import pinecone_ingest
import pinecone_search
import random



# represents the three different embedding models used 
embedding_models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "Granite": "granite-embedding",
    "Nomic-Embed": "nomic-embed-text"

}

# represents the three vector databases 
llms = {"Mistral": "mistral:latest", "Llama": "llama2:7b"}

chunk_sizes = [200, 500, 1000]
chunk_overlaps = [0, 50, 100]

test_queries = ["What is an AVL tree?", "When are Professor Fontenot's Office Hours", "How do you create a hash map?"]
csv_path = "./test_data.csv" 




def measure_time_and_memory(test_query, ingest_file, search_file, embed_model, llm, chunk_size, chunk_overlap, use_llama):

    ingest_file.run_ingest(chunk_size, chunk_overlap, embed_model, use_llama)

    tracemalloc.start() 
    start_time = time.perf_counter()

    response = search_file.run_search(test_query, embed_model, use_llama, llm)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory() 
    tracemalloc.stop() 

    execution_time = end_time - start_time
    memory_used = peak / (1024 * 1024) 
    
    print(execution_time, memory_used, response)
    return execution_time, memory_used, response


# Sampling Combos
possible_combinations = []
for embedding_name, embedding_model in embedding_models.items():
    for llm_name, llm in llms.items():
        for size in chunk_sizes:
            for overlap in chunk_overlaps:
                for query in test_queries:
                    possible_combinations.append((embedding_name, embedding_model, llm_name, llm, size, overlap, query))

sampled_combos = random.sample(possible_combinations, 4)


# Redis For Loop
'''
redis_results = []
for embedding_name, embedding_model, llm_name, llm, size, overlap, query in sampled_combos:
    if embedding_name == "Nomic-Embed":
        query_time, query_memory, response = measure_time_and_memory(
            query, redis_ingest, redis_search, embedding_model, llm, size, overlap, True
        )
    else: 
        query_time, query_memory, response = measure_time_and_memory(
            query, redis_ingest, redis_search, embedding_model, llm, size, overlap, False
        )

    redis_results.append([
        embedding_name, "Redis", size, overlap, query_time, query_memory, response, llm_name
    ])

df_redis = pd.DataFrame(redis_results, columns=[
    'embedding_model', 'Vector_DB', 'chunk_size', 'chunk_overlap',
    'query_time', 'query_memory', 'response', 'llm'])


df_redis.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))'
'''

'''
chroma_results = []
for embedding_name, embedding_model, llm_name, llm, size, overlap, query in sampled_combos:
    if embedding_name == "Nomic-Embed":
        query_time, query_memory, response = measure_time_and_memory(
            query, chroma_ingest, chroma_search, embedding_model, llm, size, overlap, True
        )
    else: 
        query_time, query_memory, response = measure_time_and_memory(
            query, chroma_ingest, chroma_search, embedding_model, llm, size, overlap, False
        )

    chroma_results.append([
        embedding_name, "Chroma", size, overlap, query_time, query_memory, response, llm_name
    ])

df_chroma = pd.DataFrame(chroma_results, columns=[
    'embedding_model', 'Vector_DB', 'chunk_size', 'chunk_overlap',
    'query_time', 'query_memory', 'response', 'llm'])

df_chroma.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
'''


pinecone_results = []
for embedding_name, embedding_model, llm_name, llm, size, overlap, query in sampled_combos:
    if embedding_name == "Nomic-Embed":
        query_time, query_memory, response = measure_time_and_memory(
            query, pinecone_ingest, pinecone_search, embedding_model, llm, size, overlap, True
        )
    else: 
        query_time, query_memory, response = measure_time_and_memory(
            query, pinecone_ingest, pinecone_search, embedding_model, llm, size, overlap, False
        )

    pinecone_results.append([
        embedding_name, "Chroma", size, overlap, query_time, query_memory, response, llm_name
    ])

df_chroma = pd.DataFrame(pinecone_ingest, columns=[
    'embedding_model', 'Vector_DB', 'chunk_size', 'chunk_overlap',
    'query_time', 'query_memory', 'response', 'llm'])

df_chroma.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))



'''
added_results = []

query_time, query_memory, response = measure_time_and_memory("How do you create a hash map?", redis_ingest, redis_search, "granite-embedding", "mistral:latest", 200, 0, True)

added_results.append([
        "Granite", "Redis", 200, 0, query_time, query_memory, response, "Llama"
    ])
df_add = pd.DataFrame(added_results, columns=[
    'embedding_model', 'Vector_DB', 'chunk_size', 'chunk_overlap',
    'query_time', 'query_memory', 'response', 'llm'])


df_add.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))'
'''
