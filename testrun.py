import os
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "predusk"

# List existing indexes
indexes = pinecone.list_indexes()
print("Existing indexes:", indexes)

if index_name in indexes:
    print(f"Index '{index_name}' exists.")
else:
    print(f"Index '{index_name}' does not exist.")
