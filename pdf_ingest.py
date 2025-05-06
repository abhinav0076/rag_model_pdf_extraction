import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

def extract_text_with_tables(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            with pdfplumber.open(filepath) as pdf:
                full_text = ""
                for page in pdf.pages:
                    # Extract text
                    text = page.extract_text() or ""
                    full_text += text + "\n"

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(str(cell).strip() for cell in row if cell)
                            full_text += row_text + "\n"

            documents.append(Document(page_content=full_text, metadata={"source": filename}))
    return documents

def create_vector_store(documents, persist_path="data/faiss_index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(persist_path)
    return persist_path

if __name__ == "__main__":
    docs = extract_text_with_tables("documents")
    create_vector_store(docs)