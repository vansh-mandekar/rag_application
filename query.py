import os
from transformers import pipeline
from utils import generate_embedding, index

# Hugging Face API key from Spaces Secrets
HF_API_KEY = os.environ.get("HF_API_KEY")
MODEL_ID = "deepset/roberta-base-squad2"  # or any other model

# Initialize Hugging Face QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    use_auth_token=HF_API_KEY
)

def get_answer(user_query, context):
    try:
        result = qa_pipeline(question=user_query, context=context)
        return result.get('answer', "Sorry, I cannot generate an answer right now.")
    except Exception as e:
        print("Error during inference:", e)
        return "Sorry, I cannot generate an answer right now."

def query_rag(user_query, story_text, top_k=3):
    """
    RAG query: retrieves relevant chunks from Pinecone and returns LLM answer with citations.
    """
    query_vector = generate_embedding(user_query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    retrieved_chunks = []
    citations = []

    for i, match in enumerate(results.get('matches', [])):
        text = match['metadata'].get('text', '')
        retrieved_chunks.append(text)
        citations.append(f"[{i+1}] {match['metadata'].get('source', '')}")

    # If no chunks retrieved, fallback to full story
    context = "\n".join(retrieved_chunks) if retrieved_chunks else story_text

    answer_text = get_answer(user_query, context)

    return answer_text, citations
