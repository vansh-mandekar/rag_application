-->Providers Used
Vector DB	Pinecone
Embeddings	Sentence-Transformers / Hugging Face
Reranker & QA	Hugging Face Transformers (roberta-base)
Frontend	Streamlit
Hosting	        Streamlit  cloud (Free)
Utilities	Numpy, Pandas, Tiktoken, Regex

-->architecture diagram
<img width="1683" height="662" alt="Screenshot 2025-09-15 133603" src="https://github.com/user-attachments/assets/5e9c0bb4-622d-425c-a8bd-e4ed46ed9d6f" />


-->what each code file does:

streamlit_app.py
|--Starts the web app using Streamlit
|--Lets users upload or paste a story
|--Lets users ask questions about the story
|--Calls functions from main.py to save the story and query.py to get answers
|--Displays the answer and related citations in the app
|--Loads environment variables like API keys and sets up the app

utils.py
|--Loads the Pinecone index and transformer model
|--Has functions to process text and generate embeddings
|--Splits the story into smaller chunks
|--Prepares text for searching and matching

query.py
|--Uses Hugging Face’s model to answer questions
|--Finds relevant story chunks from Pinecone
|--Combines the chunks and generates an answer
|--Returns the answer and citations to the app

main.py
|--Splits the story into chunks and generates embeddings
|--Saves chunks into Pinecone with extra details like source and section
|--Provides a function that the app can call to index the story

-->API's:
the api's that were used are mentioned in .env.example file, the api while deploying are saved in secrets of streamlit cloud

-->remarks:
-haven’t implemented timing or token/cost estimation.
-using Hugging Face’s question-answering pipeline instead of a separate reranker step.

-->Q/A pairs:

1. Q-What did Percy pack for his adventure?
   A-frozen fish and seaweed

2. Q-Who did Percy meet in the hidden valley?
   A-albert

3. Q-What did Percy learn from Albert?
   A-the currents of the ocean and the secret paths that connected distant lands

4. Q-Where did Percy go on his adventure?
   A-To explore beyond the icy plains.

5. Q-What was the name given to Percy after his journey?
   A-Percy the Curious Penguin.

-->Minimal Evaluation – 5 Q/A pairs

I tested the app using the story about Percy’s adventure(refer file sample_story.txt) and asked 5 questions related to key details in the story. The app was able to correctly retrieve and answer all 5 questions with accurate citations.

Success Rate:
✔ 5 out of 5 questions answered correctly
Precision: 100% – every retrieved answer was relevant and correct.
Recall: 100% – all the expected answers were retrieved and displayed by the app.

Sometimes, instead of expanding on the answer, it just gives short or one-word answers. For example, it gave “albert” instead of “a wise old albatross named Albert.”

This shows that the app is working well for this story and can retrieve relevant information from the indexed text.

-->Index Configuration

Index Name: predusk
Metric: cosine (used to measure how similar two vectors are)
Dimensions: 384 (based on the embedding model used)
Cloud Host: AWS (us-east-1 region) – the database is hosted on Amazon Web Services
Type: Dense – stores vector data in a compact format
Capacity Mode: Serverless – automatically handles scaling based on usage
Model Used for Embeddings: llama-text-embed-v2 from Pinecone
Records: 2 (sample data used for testing)
Namespace: __default__


resume link: https://drive.google.com/drive/folders/1JefYLEru0zfdllzxoS4HOm3s9EzkiZ5e?usp=sharing

