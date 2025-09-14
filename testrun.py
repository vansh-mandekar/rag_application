import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
index_name = "predusk"

indexes = pc.list_indexes().names()
print("Existing indexes:", indexes)

if index_name in indexes:
    print(f"Index '{index_name}' exists.")
else:
    print(f"Index '{index_name}' does not exist.")
