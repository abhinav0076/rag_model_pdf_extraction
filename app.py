import streamlit as st
from query_engine import ask_question_return_csv
import pandas as pd

st.set_page_config(page_title="RAG Model: Basel III PDF Query", layout="centered")
st.title("📊 RAG Model: Basel III PDF Query")

query = st.text_input("Ask a question about the financial PDFs:")
submit = st.button("Get Answer in CSV")

if submit and query:
    with st.spinner("🔍 Querying the vector database..."):
        csv_path = ask_question_return_csv(query)
        df = pd.read_csv(csv_path)

        # Show result in app
        st.subheader("Answer")
        st.write(df["Answer"].iloc[0])

        # Allow CSV download
        with open(csv_path, "rb") as file:
            st.download_button(
                label="📥 Download CSV",
                data=file,
                file_name="answer.csv",
                mime="text/csv"
            )