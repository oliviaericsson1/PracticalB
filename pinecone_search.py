from pinecone_both import generate_rag_response

def run_search(query, embed_model, use_llama, llm):
    """Wrapper function for Pinecone search"""
    return generate_rag_response(query)
