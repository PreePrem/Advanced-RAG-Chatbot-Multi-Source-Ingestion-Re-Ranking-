# Advanced-RAG-Chatbot-Multi-Source-Ingestion-Re-Ranking-


# 🧠 Advanced RAG Chatbot — Multi-Source Ingestion + Re-Ranking + Chat History

This project is an **upgraded Retrieval-Augmented Generation (RAG)** chatbot built with LangChain and powered by Groq's blazing fast LLaMA3 model.

---

## 🔥 Features

- 🔍 Multi-source ingestion (PDF, URLs, and folders)
- 🧹 HTML text cleaning & formatting normalization
- 🧠 Semantic chunking with `RecursiveCharacterTextSplitter`
- 🧾 Re-ranking with `ContextualCompressionRetriever`
- 💬 Chat history preservation across questions
- 🧰 Exception handling for missing content, chunking failures, and broken URLs
- 🧪 Easy to switch between different vector search techniques (default: FAISS)

---

## 🚀 Tech Stack

| Component        | Tool                         |
|------------------|------------------------------|
| LLM              | LLaMA 3 (via Groq API)       |
| Orchestration    | LangChain                    |
| Embeddings       | HuggingFace Sentence Transformers |
| Vector Store     | FAISS                        |
| Frontend         | Streamlit                    |
| Format Handling  | BeautifulSoup, PyPDFLoader   |

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
