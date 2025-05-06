# import os
# os.environ["STREAMLIT_WATCH_DIR"] = "."

# import streamlit as st
# from query_engine import ask_question_return_csv
# import pandas as pd

# st.set_page_config(page_title="RAG Model: Basel III PDF Query", layout="centered")
# st.title("📊 RAG Model: Basel III PDF Query")

# query = st.text_input("Ask a question about the financial PDFs:")
# submit = st.button("Get Answer in CSV")

# if submit and query:
#     with st.spinner("🔍 Querying the vector database..."):
#         csv_path = ask_question_return_csv(query)
#         df = pd.read_csv(csv_path)

#         # Show result in app
#         st.subheader("Answer")
#         st.write(df["Answer"].iloc[0])

#         # Allow CSV download
#         with open(csv_path, "rb") as file:
#             st.download_button(
#                 label="📥 Download CSV",
#                 data=file,
#                 file_name="answer.csv",
#                 mime="text/csv"
#             )


import os
import streamlit as st
import pandas as pd
import tempfile
from query_engine import ask_question_return_csv
from pdf_ingest import extract_text_with_tables, create_vector_store

# Prevent inotify overflow
os.environ["STREAMLIT_WATCH_DIR"] = "."

st.set_page_config(page_title="RAG Model: Basel III PDF Query", layout="centered")
st.title("📊 RAG Model: Basel III PDF Query")

# --- Upload Section ---
uploaded_file = st.sidebar.file_uploader(
    "Upload a Basel III PDF (Max 10MB)",
    type=["pdf"],
    accept_multiple_files=False
)

# --- Upload validation ---
if uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.sidebar.error("File is too large! Max allowed size is 10MB.")
        uploaded_file = None
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process the uploaded file in memory
            with st.spinner("🔄 Indexing uploaded file..."):
                docs = extract_text_with_tables(tmpdir)
                create_vector_store(docs, persist_path="data/temp_faiss")
            st.sidebar.success("✅ File indexed successfully!")

# --- Query Interface ---
query = st.text_input("Ask a question about the uploaded PDF:")
submit = st.button("Get Answer in CSV")

if submit and query:
    with st.spinner("🔍 Searching..."):
        csv_path = ask_question_return_csv(query, index_path="data/temp_faiss")
        df = pd.read_csv(csv_path)

        st.subheader("Answer")
        st.write(df["Answer"].iloc[0])

        with open(csv_path, "rb") as file:
            st.download_button("📥 Download CSV", data=file, file_name="answer.csv", mime="text/csv")
