import streamlit as st
from query import query_rag
from index_story import index_story  # updated import
import os

st.title("Hello from Render!")
st.write("This app is live!")

# Set a writable config directory for Streamlit
os.environ["STREAMLIT_HOME"] = os.getcwd()
os.environ["STREAMLIT_CONFIG_DIR"] = os.getcwd()

# Load environment variables if using a .env file (optional on Render)
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
HF_API_KEY = os.getenv("HF_API_KEY")

# Optional: Initialize Pinecone if needed
# from pinecone import Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

st.set_page_config(page_title="Predusk RAG App", layout="wide")
st.title("📖 Predusk RAG App")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.write("Using environment variables set in Hugging Face Spaces Secrets.")

# Story input
st.subheader("Upload or Paste Story Text")
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
story_text = ""

if uploaded_file is not None:
    story_text = uploaded_file.read().decode("utf-8")
else:
    story_text = st.text_area("Or paste text here", height=200)

# Index story button
if story_text.strip() != "":
    if st.button("Index Story"):
        index_story(story_text)
        st.success("Story indexed successfully!")

# Query section
st.subheader("Ask a Question")
user_query = st.text_input("Enter your question:")

if user_query and story_text.strip() != "":
    if st.button("Get Answer"):
        with st.spinner("Retrieving answer..."):
            answer, citations = query_rag(user_query, story_text)
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Citations:")
            for c in citations:
                st.write(c)

# Debugging environment variables
import time
time.sleep(5)  # Delay so logs can be viewed easily

# print("Checking environment variables:")
# print("PINECONE_API_KEY:", PINECONE_API_KEY)
# print("PINECONE_ENVIRONMENT:", PINECONE_ENVIRONMENT)
# print("HF_API_KEY:", HF_API_KEY)

