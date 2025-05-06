# query_engine.py

import os
import csv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

def get_qa_chain(index_path: str):
    """
    Load FAISS vector store and initialize RetrievalQA chain using OpenAI.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_key), retriever=retriever)
    return qa_chain

def ask_question_return_csv(query: str, index_path: str):
    """
    Run a QA query using the index, write the result to a CSV, and return the path.
    """
    chain = get_qa_chain(index_path)
    result = chain.invoke({"query": query})

    # Save result to CSV
    csv_path = os.path.join(index_path, "result.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Answer"])
        writer.writerow([query, result["result"]])

    return csv_path