import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import chromadb
from chromadb import HttpClient

# Connect to ChromaDB running in Docker
chroma_client = HttpClient(host="localhost", port=8000)
COLLECTION_NAME = "embedding_collection"
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
    if use_llama:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        sentence_transformer = SentenceTransformer(model)
        return sentence_transformer.encode(text).tolist()


def search_embeddings(query, model, use_llama=False, top_k=3):
    try:
        query_embedding = get_embedding(query, model, use_llama)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        top_results = []
        for doc, distance, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
            top_results.append({
                "file": meta.get("file", "Unknown"),
                "page": meta.get("page", "Unknown"),
                "chunk": meta.get("chunk_index", "Unknown"),
                "similarity": distance,
                "text": doc,
            })

        # Print for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.4f}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, model, context_results):
    # Construct context
    context_str = "\n\n".join(
        [
            f"From {res['file']} (page {res['page']}, chunk {res['chunk']}):\n{res['text']}"
            for res in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive RAG loop"""
    print("🔍 RAG Search Interface (ChromaDB)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search in ChromaDB
        context_results = search_embeddings(query, "all-MiniLM-L6-v2", use_llama=False)

        # Generate LLM response
        response = generate_rag_response(query, "llama3.2:latest", context_results)

        print("\n--- Response ---")
        print(response)


def run_search(query, embedding_model="all-MiniLM-L6-v2", use_llama=False, llm_model="llama3.2:latest"):
    context_results = search_embeddings(query, embedding_model, use_llama)
    response = generate_rag_response(query, llm_model, context_results)
    return response


if __name__ == "__main__":
    print("Running ChromaDB RAG Search")
    interactive_search()
