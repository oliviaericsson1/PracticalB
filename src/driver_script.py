import ollama
import chrombadb
from sentence_transformers import SentenceTransformer
import redis
from redis.commands.search.query import Query

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

CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAPS = [0, 50, 100]

llms = ["mistral:latest", "llama2:7b"]

file = "performance.csv"

# memory function  
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def run_experiment(model_name, db_type, llm, chunk_size, chunk_overlap):
    print(f"\nRunning Experiment: Model={model_name}, DB={db_type}, LLM={llm}, Chunk Size={chunk_size}, Overlap={chunk_overlap}")

    start_time = time.time()
    start_memory = get_memory_usage()
    subprocess.run(["python", "ingest.py"])  # Run ingestion script

    end_memory = get_memory_usage()
    end_time = time.time()

    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    return {
        "Embedding Model": model_name,
        "Vector DB": db_type,
        "LLM": llm,
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        "Time (s)": elapsed_time,
        "Memory (MB)": memory_used
    }
results = []
for model in embedding_models:
    for db in ["Redis", "Chroma"]:
        for llm in llms:
            for size in CHUNK_SIZES:
                for overlap in CHUNK_OVERLAPS:
                    result = run_experiment(model, db, llm, size, overlap)
                    results.append(result)


df = pd.DataFrame(results)
df.to_csv(file, index=False)

'''
for model in embedding_models:
  for db in dbs:
    for llm in llms:
      for size in chunk_sizes:
      
        # add in ingest functions as necessary
        # add in search as necessary with model, db, llm inputs
        # Store time and memory data and append it to some csv
        '''
