
# Embedding Models

# Vector DBs
redis_client = redis.Redis(host="localhost", port=6379, db=0) # Redis 
chroma_client = chromadb.PersistentClient(path="./chroma_db") # Chroma
# 1 Additional vector database
