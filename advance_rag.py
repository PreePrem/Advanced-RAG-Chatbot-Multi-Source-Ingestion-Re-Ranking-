import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import tempfile
import shutil
import html2text

# Set up the Streamlit app
st.set_page_config(page_title="🧠 Multi-Source RAG Chatbot", layout="wide")
st.title("🔍 RAG Chatbot with PDFs, URLs, and Directories")

GROQ_API_KEY = "load-your-personal-groq-api-key"

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# File uploader
pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
urls = st.text_area("Enter URLs (comma separated)")
folder_path = st.text_input("Path to directory (optional)")

# Load documents
docs = []
if pdfs:
    for pdf in pdfs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
if urls:
    url_list = [u.strip() for u in urls.split(",")]
    loader = UnstructuredURLLoader(urls=url_list)
    try:
        raw_docs = loader.load()
        # Convert HTML to clean text
        parser = html2text.HTML2Text()
        for d in raw_docs:
            d.page_content = parser.handle(d.page_content)
        docs.extend(raw_docs)
    except Exception as e:
        st.warning(f"Failed to load URL content: {e}")
if folder_path and os.path.isdir(folder_path):
    try:
        loader = DirectoryLoader(folder_path)
        docs.extend(loader.load())
    except Exception as e:
        st.warning(f"Error reading directory: {e}")

if docs:
    st.success("✅ Documents Loaded")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # Retriever with re-ranking
    base_retriever = db.as_retriever(search_kwargs={"k": 6})
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    # Prompt
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are an assistant for Q&A. Use the context to answer the question below.
Answer in under 3 sentences. Be precise.
Question: {question}
Context: {context}
Answer:
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Your question")

    if query:
        with st.spinner("Thinking..."):
            result = chain(query)

            st.markdown("#### 🧠 Answer")
            st.write(result["result"])

            st.markdown("#### 📚 Source Chunks")
            for doc in result["source_documents"]:
                st.info(doc.page_content[:300] + "...")
else:
    st.info("📄 Upload documents or provide URLs/directories to begin.")
