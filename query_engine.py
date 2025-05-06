import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pdf_ingest import extract_text_with_tables, create_vector_store

load_dotenv()

INDEX_PATH = "data/faiss_index"

def get_qa_chain(index_path=INDEX_PATH):
    embeddings = OpenAIEmbeddings()

    # If index not found, recreate it from documents
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        docs = extract_text_with_tables("documents")
        create_vector_store(docs, persist_path=index_path)

    vector_store = FAISS.load_local(index_path, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

def ask_question_return_csv(question, output_csv="data/output.csv"):
    chain = get_qa_chain()
    response = chain.run(question)

    df = pd.DataFrame([[question, response]], columns=["Question", "Answer"])
    df.to_csv(output_csv, index=False)
    return output_csv
