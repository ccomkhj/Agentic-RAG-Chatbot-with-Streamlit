import yaml
import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
    TextLoader,
    CSVLoader
)
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
import google.generativeai as genai

# Load credentials from YAML file
def load_credentials():
    try:
        with open("credentials.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Credentials file not found. Please create credentials.yaml file.")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing credentials YAML: {str(e)}")
        return {}
    except IOError as e:
        st.error(f"I/O error reading credentials: {str(e)}")
        return {}

# Function to get the appropriate document loader based on file extension
def get_document_loader(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension in [".docx", ".doc"]:
        return Docx2txtLoader(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path)
    elif file_extension in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path)
    elif file_extension in [".txt", ".md"]:
        return TextLoader(file_path)
    elif file_extension == ".csv":
        return CSVLoader(file_path)
    else:
        # Default loader for other file types
        return UnstructuredFileLoader(file_path)

# Process uploaded files for RAG
def process_files(uploaded_files):
    # Get credentials
    creds = load_credentials()
    
    # Configure API key and project ID
    if "API_KEY" in creds and creds["API_KEY"]:
        api_key = creds["API_KEY"]
    else:
        st.error("Error: API Key not found in credentials.yaml")
        return None
        
    if "PROJECT_ID" in creds and creds["PROJECT_ID"]:
        project_id = creds["PROJECT_ID"]
    else:
        st.error("Error: PROJECT_ID not found in credentials.yaml")
        return None
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    if not uploaded_files:
        return None
    
    # Process files
    all_docs = []
    with st.spinner("Processing files..."):
        for uploaded_file in uploaded_files:
            # Create a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Get appropriate loader and load document
                loader = get_document_loader(temp_path)
                documents = loader.load()
                
                # Add metadata about source
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                
                all_docs.extend(documents)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temp files
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except:
                    pass
    
    if not all_docs:
        st.warning("No documents were successfully processed.")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    
    # Create embedding model
    embeddings = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest",
        project=project_id
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store
