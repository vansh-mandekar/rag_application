import tiktoken
import os
import numpy as np
import Pinecone
from utils import chunk_text, generate_embedding

# Hugging Face Spaces secrets
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # optional

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "predusk"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # same as embedding size in generate_embedding()
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # or use your PINECONE_ENVIRONMENT if needed
        )
    )

index = pc.Index(index_name)

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
            "text": chunk   # Store the actual chunk for retrieval
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

