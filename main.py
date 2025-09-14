import os
import numpy as np
from pinecone import Pinecone  # import the class with a capital P
from utils import chunk_text, generate_embedding

# Load Pinecone API key and environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # optional

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "predusk"
index = pc.Index(index_name)  # access existing index

# Function to index story text
def index_story(story_text):
    chunks = chunk_text(story_text)
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        vector = generate_embedding(chunk)
        metadata = {
            "source": "user_input",
            "title": "User Provided Story",
            "section": f"paragraph_{i+1}",
            "position": i,
            "text": chunk
        }
        index.upsert(vectors=[(f"user_{i}", vector, metadata)])

    print("All chunks upserted to Pinecone!")

# Run only when executed directly
if __name__ == "__main__":
    sample_file = "sample_story.txt"
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            story_text = f.read()
        index_story(story_text)
    else:
        print(f"{sample_file} not found. Use `index_story(story_text)` for custom stories.")
