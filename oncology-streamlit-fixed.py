import streamlit as st
import pandas as pd
import time
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# Set page config
st.set_page_config(
    page_title="Oncology Guidelines Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    .sub-header {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }
    .source-box {
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    .chat-user {
        background-color: #DBEAFE;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-ai {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .relevance-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Class for oncology retriever tool with real document retrieval
class OncologyRetriever:
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings
        self.loaded = False
    
    def load_documents(self, uploaded_files):
        try:
            st.info("Processing uploaded documents... This may take a moment.")
            temp_dir = tempfile.mkdtemp()
            all_docs = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary directory
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document based on file type
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:  # Default to text loader for other formats
                    loader = TextLoader(file_path)
                    
                docs = loader.load()
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                
                all_docs.extend(docs)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                add_start_index=True,
                strip_whitespace=True,
            )
            docs_processed = text_splitter.split_documents(all_docs)
            
            # Assign IDs to documents
            for idx, doc in enumerate(docs_processed):
                doc.metadata["id"] = f"doc_{idx}"
            
            # Create retriever based on preference
            if self.use_embeddings:
                # Use HuggingFace embeddings with FAISS
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.from_documents(docs_processed, embeddings)
                self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            else:
                # Use simpler BM25 retriever
                self.retriever = BM25Retriever.from_documents(docs_processed, k=5)
            
            st.session_state.docs_processed = True
            st.session_state.retriever = self.retriever
            
            st.success(f"Successfully processed {len(docs_processed)} document chunks!")
            return True
            
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return False
    
    def query(self, question):
        if not st.session_state.docs_processed:
            return {
                "answer": "Please upload documents first to enable the retrieval system.",
                "sources": []
            }
        
        try:
            # Retrieve relevant documents
            with st.spinner("Searching through oncology guidelines..."):
                retrieved_docs = self.retriever.get_relevant_documents(question)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find specific information about this query in the uploaded documents. Please try a different question or upload additional relevant documents.",
                    "sources": []
                }
            
            # Calculate relevance scores (simplified version)
            # In real implementation, you might use actual relevance scores from the retriever
            max_score = 98
            sources = []
            
            for i, doc in enumerate(retrieved_docs):
                # Simulate descending relevance scores
                relevance = max(70, max_score - (i * 3))
                source_info = {
                    "title": doc.metadata.get("source", "Unknown Source"),
                    "content": doc.page_content,
                    "relevance": relevance
                }
                sources.append(source_info)
            
            # For a simple implementation, we'll just concat relevant info
            # In a real system, you might want to use an LLM to synthesize an answer
            answer = f"Based on the retrieved documents, here's what I found about '{question}':\n\n"
            answer += sources[0]["content"]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error during retrieval: {str(e)}",
                "sources": []
            }

# Create sidebar with options
def create_sidebar():
    st.sidebar.markdown("<h1 class='main-header'>Oncology Assistant</h1>", unsafe_allow_html=True)
    st.sidebar.write("Powered by AI and evidence-based cancer guidelines")
    
    # Document uploader
    st.sidebar.markdown("<h2 class='sub-header'>Upload Guidelines</h2>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader(
        "Upload oncology documents (PDF, TXT)", 
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    use_embeddings = st.sidebar.checkbox("Use embeddings for better retrieval", value=True)
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        retriever = OncologyRetriever(use_embeddings=use_embeddings)
        retriever.load_documents(uploaded_files)
    
    st.sidebar.markdown("<h2 class='sub-header'>Cancer Types</h2>", unsafe_allow_html=True)
    cancer_types = ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer", 
                    "Melanoma", "Lymphoma", "Leukemia", "Ovarian Cancer"]
    
    for cancer in cancer_types:
        if st.sidebar.button(cancer, key=f"cancer_{cancer}"):
            query = f"What are the current treatment guidelines for {cancer}?"
            st.session_state.chat_history.append({"role": "user", "content": query})
            process_query(query)
    
    st.sidebar.markdown("<h2 class='sub-header'>Clinical Topics</h2>", unsafe_allow_html=True)
    topics = ["Staging", "Treatment Guidelines", "Chemotherapy Regimens", 
              "Immunotherapy", "Radiation Protocols", "Surgical Approaches"]
    
    for topic in topics:
        if st.sidebar.button(topic, key=f"topic_{topic}"):
            query = f"What are the latest guidelines regarding {topic.lower()}?"
            st.session_state.chat_history.append({"role": "user", "content": query})
            process_query(query)

# Process the query and update chat history
def process_query(query):
    if not st.session_state.docs_processed:
        st.warning("Please upload and process oncology guidelines documents first.")
        result = {
            "answer": "I need oncology guidelines documents to provide accurate information. Please upload some documents using the sidebar uploader.",
            "sources": []
        }
    else:
        # Create a temporary retriever that uses the session state's retriever
        retriever = OncologyRetriever()
        retriever.loaded = True
        retriever.retriever = st.session_state.retriever
        
        with st.spinner("Searching oncology guidelines..."):
            result = retriever.query(query)
    
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", [])})

# Main application layout
def main():
    # Layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 class='main-header'>Oncology Guidelines Assistant</h1>", unsafe_allow_html=True)
        st.write("Ask any question about cancer diagnosis, staging, or treatment guidelines.")
        
        # Upload status indicator
        if not st.session_state.docs_processed:
            st.warning("No documents have been processed. Please upload oncology guidelines documents using the sidebar uploader.")
        else:
            st.success("Documents processed successfully! You can now ask questions about the uploaded guidelines.")
        
        # Chat interface
        st.markdown("<h2 class='sub-header'>Chat Interface</h2>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.markdown(f"<div class='source-box'><strong>{source['title']}</strong><br>{source['content']}<br><span class='relevance-badge'>Relevance: {source['relevance']}%</span></div>", unsafe_allow_html=True)
        
        # Query input
        query = st.text_input("Ask about cancer guidelines:", key="query_input")
        if st.button("Submit", key="submit_button"):
            if query:
                st.session_state.chat_history.append({"role": "user", "content": query})
                process_query(query)
                # Clear the input box after submission by rerunning the app
                st.experimental_rerun()
    
    with col2:
        st.markdown("<h2 class='sub-header'>Document Statistics</h2>", unsafe_allow_html=True)
        
        if st.session_state.docs_processed:
            st.info("Document processing complete")
            
            # Get document sources
            sources = set()
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "assistant" and "sources" in message:
                        for source in message["sources"]:
                            sources.add(source["title"])
            
            # Display basic stats
            st.metric("Documents Loaded", len(sources) if sources else "N/A")
            
            # Show list of document sources
            if sources:
                st.write("Loaded documents:")
                for source in sources:
                    st.write(f"- {source}")
            
        else:
            st.warning("No documents processed yet")
            st.markdown("""
            ### How to use this assistant:
            1. Upload oncology guidelines documents using the sidebar uploader
            2. Click "Process Documents" to analyze and index the content
            3. Ask questions about cancer treatments, diagnosis, or guidelines
            
            You can upload NCCN, ASCO, or other oncology guidelines in PDF or text format.
            """)

# Create the sidebar
create_sidebar()

# Run the main application
main()
