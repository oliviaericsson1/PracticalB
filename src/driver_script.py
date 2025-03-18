
# Embedding Models
embedding_models = {"MiniLM": SentenceTransformer("all-MiniLM-L6-v2"), 
                    "MPNet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"), 
                    "InstructorXL": SentenceTransformer("hkunlp/instructor-xl")
                   }

# Vector DBs
redis_client = redis.Redis(host="localhost", port=6379, db=0) # Redis 
chroma_client = chromadb.PersistentClient(path="./chroma_db") # Chroma
# 1 Additional vector database
