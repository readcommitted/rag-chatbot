## rag-chatbot (Pinecone + OpenAI + Streamlit)
A lightweight, modular Streamlit app demonstrating the capabilities of a Retrieval-Augmented Generation (RAG) pipeline. This codebase integrates **Pinecone** for vector similarity search, **OpenAI** for language generation (e.g. GPT-4), and **Streamlit** for an interactive UI — making it ideal for showcasing semantic search and grounded generation with real-world data.

## see it in action
- **RAG Chatbot** - https://chatbot-readcommitted.streamlit.app
---

## 🔧 Tech Stack

- **Pinecone** – Vector search engine for efficient embedding retrieval
- **OpenAI** – GPT-4 (or GPT-3.5) for generating grounded, context-aware responses
- **Streamlit** – UI for interacting with the RAG system
- **Sentence Transformers** – `all-MiniLM-L6-v2` model for 384-dimensional embeddings
- **Docker + Docker Compose** – For containerized, portable deployment

---

## 🚀 Features

- 🔍 **Semantic vector search** with confidence scoring  
- 🧠 **RAG pipeline** for GPT-4 grounded responses  
- 🎚️ Adjustable **similarity threshold** (via slider)  
- 📄 **Chunk-level source metadata** display  
- 📊 **Top-K retrieval** with semantic scoring  
- ⚙️ **Developer Mode**: inspect prompts, payloads, and intermediate steps  
- 📦 Fully **containerized** setup for local or cloud environments  
- 🗺️ Embedded **architecture diagram** for easy understanding  

---

## 📌 Notes

- This repository **does not include the pipeline** to preprocess documents, generate embeddings, or populate the vector database.
- It assumes your **Pinecone index is already populated** with 384-dimensional sentence embeddings (e.g., from the `all-MiniLM-L6-v2` model).
- If you need ingestion functionality, consider implementing a separate data pipeline to:
  - Load and chunk documents
  - Generate sentence embeddings
  - Index them into Pinecone
- The embedding model (e.g., all-MiniLM-L6-v2) is loaded into memory during startup.
- For improved semantic precision and lower memory usage, the app can be reconfigured to use API-based embedding (e.g., OpenAI's text-embedding-ada-002) instead of loading local models at runtime.

---
## 📦 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/readcommitted/rag-chatbot.git
cd rag-chatbot
