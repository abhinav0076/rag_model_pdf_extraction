import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def get_qa_chain(index_path="data/faiss_index"):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

def ask_question_return_csv(question, output_csv="data/output.csv"):
    chain = get_qa_chain()
    response = chain.run(question)

    # Save result to CSV
    df = pd.DataFrame([[question, response]], columns=["Question", "Answer"])
    df.to_csv(output_csv, index=False)
    return output_csv