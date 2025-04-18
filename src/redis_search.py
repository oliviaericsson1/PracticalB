import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField


# initializes the model 
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
VECTOR_DIM = 784
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
    '''
    Takes in text, embedding model, and use_llama boolean, 
    Generate an embedding based on either a llama embedding model or SentenceTransformer
    '''
    if use_llama == True:
        response = ollama.embeddings(model=model, prompt=text)
        response["embedding"]
    else: 
        sentence_transformer = SentenceTransformer(model)
        response = sentence_transformer.encode(text).tolist()
        return response

def search_embeddings(query, model, use_llama = False, top_k=3):
    '''
    Takes in a query, model, use_llama boolean, and a top_k value, 
    Searches through the collection for the top_k most similar text chunks based on the query
    '''
    query_embedding = get_embedding(query, model, use_llama)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, model, context_results):
    '''
    Interactive search experience for test-taker
    '''
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query, "hkunlp/instructor-xl")

        # Generate RAG response
        response = generate_rag_response(query, "llama2:7b", context_results)

        print("\n--- Response ---")
        print(response)

def run_search(query, embedding_model,use_llama=False, llm_model="llama2:7b"):
    '''
    Takes in query, embedding_model, use_llama boolean, and llm_model, 
    Runs all relevant function in search file for testing purposes
    '''
     context_results = search_embeddings(query, embedding_model, use_llama)
     response = generate_rag_response(query, llm_model, context_results)
     return response

if __name__ == "__main__":
    print("Running search")
    interactive_search()
