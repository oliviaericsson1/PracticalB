import pinecone_ingest
import pinecone_search
import time
import tracemalloc
import os
import pandas as pd
import random

# Embedding Models
embedding_models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "Nomic-Embed": "nomic-embed-text"
}

llms = {"Mistral": "mistral:latest", "Llama": "llama3.2:latest"}

chunk_sizes = [200, 500, 1000]
chunk_overlaps = [0, 50, 100]

test_queries = [
    "What is an AVL tree?",
    "When are Professor Fontenot's Office Hours",
    "How do you create a hash map?"
]
csv_path = "./pinecone_test.csv" 

def measure_time_and_memory(test_query, embed_model, llm, chunk_size, chunk_overlap, use_llama):
    """Measure performance of Pinecone operations"""
    # Run ingestion (with timing)
    tracemalloc.start()
    start_time = time.perf_counter()
    
    pinecone_ingest.run_ingest(
        model_name=embed_model,
        use_llama=use_llama,
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )
    
    # Measure search and RAG generation
    search_start = time.perf_counter()
    response = pinecone_search.run_search(
        query=test_query,
        embed_model=embed_model,
        use_llama=use_llama,
        llm=llm
    )
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time = end_time - start_time
    search_time = end_time - search_start
    memory_used = peak / (1024 * 1024)  # Convert to MB
    
    print(f"Total Time: {execution_time:.2f}s | Search Time: {search_time:.2f}s | Memory: {memory_used:.2f}MB")
    return execution_time, search_time, memory_used, response

# Generate all possible test combinations
possible_combinations = [
    (emb_name, emb_model, llm_name, llm, size, overlap, query)
    for emb_name, emb_model in embedding_models.items()
    for llm_name, llm in llms.items()
    for size in chunk_sizes
    for overlap in chunk_overlaps
    for query in test_queries
]

# Randomly sample combinations (4 for testing)
sampled_combos = random.sample(possible_combinations, min(4, len(possible_combinations)))

# Run tests
pinecone_results = []
for emb_name, emb_model, llm_name, llm, size, overlap, query in sampled_combos:
    use_llama = emb_name == "Nomic-Embed"
    
    print(f"\nTesting: {emb_name}, {llm_name}, Chunk {size}/{overlap}, Query: '{query}'")
    
    try:
        total_time, search_time, memory, response = measure_time_and_memory(
            test_query=query,
            embed_model=emb_model,
            llm=llm,
            chunk_size=size,
            chunk_overlap=overlap,
            use_llama=use_llama
        )
        
        pinecone_results.append({
            'embedding_model': emb_name,
            'llm': llm_name,
            'chunk_size': size,
            'chunk_overlap': overlap,
            'query': query,
            'total_time': total_time,
            'search_time': search_time,
            'memory_used': memory,
            'response': response
        })
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        continue

# Save results
df = pd.DataFrame(pinecone_results)
df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
print(f"\nResults saved to {csv_path}")
