import os
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
persist_directory = "chroma_db"
urls = []

# Initialize the LLM
llm = ChatGroq(
    groq_api_key="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",
    model="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=500
)

# Streamlit UI setup
st.title("News Research Tool 📰")
st.sidebar.title("Enter News Article URLs")

# Collect URLs
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

# Main placeholder
main_placeholder = st.empty()

# Session state to track processing
if "url_processed" not in st.session_state:
    st.session_state.url_processed = False

# Process URLs button
if st.sidebar.button("Process URLs"):
    if urls:
        st.session_state.url_processed = True
        main_placeholder.text("🔄 Loading data...")

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ',', '.'],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings()

        # Check if Chroma DB exists
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            main_placeholder.text("✅ Loaded existing Chroma DB")
        else:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            main_placeholder.text("✅ New Chroma DB created")

        time.sleep(1)

# Query input (only if URLs are processed)
if st.session_state.url_processed:
    query = st.text_input("Enter your query:")

    if st.sidebar.button("Search"):
        if query:
            # Load vectorstore directly from Chroma
            embeddings = HuggingFaceEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # Retrieve documents manually
            docs = retriever.get_relevant_documents(query)

            # Show retrieved documents for debugging
            st.subheader("🔎 Retrieved Context")
            for idx, doc in enumerate(docs):
                st.markdown(f"**Document {idx+1}:** {doc.page_content}")

            # Set up RetrievalQA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)

            # Display final answer
            st.header("🧠 Answer")
            st.subheader(result["answer"])

# Optional: Reset everything
if st.sidebar.button("Reset"):
    st.session_state.url_processed = False
    urls.clear()
    main_placeholder.text("🔄 Data reset. Please input new URLs.")
