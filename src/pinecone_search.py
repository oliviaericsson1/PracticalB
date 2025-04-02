import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from pinecone import Pinecone
from pineconetest import get_embedding, INDEX_NAME

# Configuring API key, vector dimensions, pinecone client, and index
PINECONE_API_KEY = "pcsk_54uT6j_TxDRZUd3qDQEPYdWaNw3PWhd3xWfWohQaXJVJWjmm8Nbxsys5DLXNYHb3W6hNL5"
VECTOR_DIM = 768
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


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


def search_embeddings(query, model, use_llama=False, top_k=3):
    '''
    Takes in a query, model, use_llama boolean, and a top_k value, 
    Searches through the collection for the top_k most similar text chunks based on the query
    '''
    try:
        query_embedding = get_embedding(query, model, use_llama)

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        ).matches

        top_results = [
            {
                "file": r.metadata.get("file", "Unknown"),
                "page": r.metadata.get("page", "Unknown"),
                "chunk": r.metadata.get("chunk_index", "Unknown"),
                "text": r.metadata.get("text", "")[:200],
                "similarity": r.score
            }
            for r in results
        ]

        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Score: {result['similarity']:.4f}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, model, context_results):
    '''
    Takes in query, model, and context_results, 
    Provides a RAG response based on the query and prompt to the model
    '''
    context_str = "\n".join(
        [
            f"From {r.get('file', 'Unknown')} (page {r.get('page', '?')}, chunk {r.get('chunk', '?')}): {r.get('text', '')}"
            for r in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query. If the context is not relevant, say "I don't know."

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search(embedding_model="nomic-embed-text", llm_model="llama3.2:latest", use_llama=True):
    '''
    Interactive search experience for test-taker
    '''
    print("🔍 Pinecone RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, embedding_model, use_llama)
        response = generate_rag_response(query, llm_model, context_results)

        print("\n--- Response ---")
        print(response)


def run_search(query, embedding_model, use_llama=False, llm_model="llama3.2:latest"):
    '''
    Takes in chunk_size, chunk_overlap, embedding model, and use_llama boolean, and runs all necessary
    functions in this ingest file for testing purposes
    '''
    context_results = search_embeddings(query, embedding_model, use_llama)
    return generate_rag_response(query, llm_model, context_results)


if __name__ == "__main__":
    interactive_search()
