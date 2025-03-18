

redis_client = redis.Redis(host="localhost", port=6379, db=0)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
