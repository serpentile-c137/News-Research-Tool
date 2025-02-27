import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
import langchain
from langchain import LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import google.generativeai as genai
from google.generativeai import GenerativeModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

genai.configure()

st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini"  # Change to use .index for FAISS

main_placeholder = st.empty()
MODEL = "gemini-1.5-flash"
# Create a new Chat with ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.9)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split the data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a file using FAISS's own methods
    vector_store.save_local(file_path)  # Save the FAISS index

    main_placeholder.text("FAISS Index Saved Successfully! ✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the FAISS index from the saved file
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)  # Load the FAISS index
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
    else:
        main_placeholder.text("FAISS index not found. Please process the URLs first.")

# footer_html = """<div style='text-align: center;'>
#   <p>Made with ❤️ by Shardul Gore</p>
# </div>"""

footer_html = """
    <div style='position: fixed; bottom: 0; text-align: center !important;'>
        <p>Made with ❤️ by Shardul Gore</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
