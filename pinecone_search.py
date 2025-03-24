import os
import ollama
from pinecone import Pinecone

# Pinecone setup
PINECONE_API_KEY = "pcsk_62H5cT_7YHFMU5JH1PhPRGjjdVRChy8QTfYNdxWj6JsSqAJgbSmCj2hZZc9pkH12h4aUUf" 
INDEX_NAME = "embedding-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get embedding for text using Ollama"""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(query, top_k=3):
    """Search Pinecone for the most relevant embeddings"""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

def generate_rag_response(query):
    """Generate response using retrieved context"""
    context_results = search_embeddings(query)
    if not context_results:
        return "No relevant context found for your query."
        
    context_str = "\n".join(
        [
            f"From {result.metadata['file']} (page {result.metadata['page']}): {result.metadata['text']}..."
            for result in context_results
        ]
    )
    
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
    
Context:
{context_str}

Question: {query}

Provide a detailed answer based on the context:"""
    
    response = ollama.chat(
        model="llama3.2:latest",  # Updated model name
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def interactive_search():
    """Interactive RAG search"""
    print("Pinecone RAG Search Interface")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
                
            print("\nProcessing your query...")
            response = generate_rag_response(query)
            print("\n--- Response ---\n")
            print(response)
            print("\n" + "-"*40)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting interactive search...")
    interactive_search()
