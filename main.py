import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("News research tool")
st.sidebar.title("News articles URLs")

for i in range(3):
    st.sidebar.text_input(f"URL {i+1}")

process_url_clicked = st.sidebar.button("Process URL's")

if process_url_clicked:
    pass