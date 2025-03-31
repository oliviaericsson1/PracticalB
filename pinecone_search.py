import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from pinecone import Pinecone
from pineconetest import get_embedding, INDEX_NAME

# Pinecone setup
PINECONE_API_KEY = "pcsk_54uT6j_TxDRZUd3qDQEPYdWaNw3PWhd3xWfWohQaXJVJWjmm8Nbxsys5DLXNYHb3W6hNL5"
VECTOR_DIM = 768
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def get_embedding(text: str, model: str, use_llama: bool = False) -> list:
    if use_llama:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        return SentenceTransformer(model).encode(text).tolist()


def search_embeddings(query, model, use_llama=False, top_k=3):
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


def interactive_search():
    """Interactive RAG loop"""
    print("🔍 RAG Search Interface (Pinecone)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        
        context_results = search_embeddings(query, "all-MiniLM-L6-v2", use_llama=False)

        # Generate LLM response
        response = generate_rag_response(query, "llama3.2:latest", context_results)

        print("\n--- Response ---")
        print(response)


def run_search(query, embedding_model, use_llama=False, llm_model="llama3.2:latest"):
    context_results = search_embeddings(query, embedding_model, use_llama)
    return generate_rag_response(query, llm_model, context_results)


if __name__ == "__main__":
    interactive_search()
