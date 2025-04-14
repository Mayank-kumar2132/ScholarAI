import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import arxiv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def fetch_arxiv_papers(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_stream = BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensure no NoneType error
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def search_vector_store(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return new_db.similarity_search(query)


def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context".
    
    Context:\n {context}?\n 
    Question:\n {question}\n 
    Answer:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    docs = search_vector_store(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("ScholarAI - Research Assistant")
    st.header("📚 ScholarAI - AI Research Assistant")

    # User Question
    user_question = st.text_input("🔍 Ask a Question from Uploaded PDFs or arXiv Papers")
    if user_question:
        user_input(user_question)

    # Sidebar for Uploading PDFs
    with st.sidebar:
        st.title("📂 Upload Research Papers")
        pdf_docs = st.file_uploader("Upload PDFs & Click Process", accept_multiple_files=True)

        if st.button("🔄 Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("✅ PDFs Processed Successfully!")

        # Search arXiv Papers
        st.subheader("📖 Search Academic Papers")
        arxiv_query = st.text_input("Enter research topic:")
        if st.button("🔍 Fetch Papers"):
            with st.spinner("Fetching papers..."):
                papers = fetch_arxiv_papers(arxiv_query)
                for paper in papers:
                    st.write(f"**{paper['title']}**")
                    st.write(paper["summary"])
                    st.write(f"[📄 Read Full Paper]({paper['pdf_url']})\n")

if __name__ == "__main__":
    main()

