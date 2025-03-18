import ollama
import chrombadb

# Embedding Models
embedding_models = {"MiniLM": SentenceTransformer("all-MiniLM-L6-v2"), 
                    "MPNet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"), 
                    "InstructorXL": SentenceTransformer("hkunlp/instructor-xl")
                   }

# Vector DBs
# Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0) 
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"

# Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="test_collection")

CHUNK_SIZES = {200, 500, 1000}
CHUNK_OVERLAPS = {0, 50, 100}

'''
for model in embedding_models:
  for db in dbs:
    for llm in llms:
      for size in chunk_sizes:
      
        # add in ingest functions as necessary
        # add in search as necessary with model, db, llm inputs
        # Store time and memory data and append it to some csv
        '''
