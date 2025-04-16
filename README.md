# 📚 Scholar AI – AI-Powered Research Assistant

**Scholar AI** is a GenAI-based research assistant that allows you to upload academic PDFs or search arXiv papers and ask questions about them. It uses powerful embeddings and language models to fetch relevant context and provide detailed answers to your research queries.

---

## 🚀 Features

- ✅ Upload and analyze multiple research paper PDFs  
- 🔍 Ask questions and get detailed AI-generated responses based on the content  
- 🧠 Uses Google Generative AI (Gemini) for both embeddings and chat-based reasoning  
- 📖 Search academic research papers from **arXiv** by topic  
- 💾 Locally stores and retrieves documents using **FAISS** for fast vector search  

---

## 🛠️ Tech Stack

- Python  
- Streamlit (for the web interface)  
- LangChain (for chaining models and prompts)  
- Google Generative AI (`gemini-pro`, `embedding-001`)  
- FAISS (for vector database)  
- PyPDF2 (to read PDF content)  
- arxiv (to search academic papers)  

---

## 📂 Project Structure

scholar-ai/ │ ├── app.py # Main Streamlit app
├── .env # Contains GOOGLE_API_KEY
├── faiss_index/ # Folder to store vector database
└── requirements.txt # Python dependencies
