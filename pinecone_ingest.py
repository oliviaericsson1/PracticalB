import os
from pinecone_both import process_pdfs, setup_pinecone_index

def run_ingest(chunk_size, overlap, model_name, use_llama):
    """Wrapper function for Pinecone ingestion"""
    setup_pinecone_index(model_name, use_llama)
    process_pdfs(
        model_name=model_name,
        use_llama=use_llama,
        chunk_size=chunk_size,
        overlap=overlap
    )
