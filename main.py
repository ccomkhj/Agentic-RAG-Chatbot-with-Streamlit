import streamlit as st
from utils import load_credentials, process_files
from llm import generate_response, set_vector_store

# Page Configuration
st.set_page_config(page_title="Gemini AI Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar with configuration options
with st.sidebar:
    st.title("Chatbot Configuration")
    
    # Load credentials
    credentials = load_credentials()
    
    # Display configuration
    st.header("Current Configuration")
    st.info(
        f"Model: {credentials.get('AGENT_MODEL_NAME', 'gemini-1.5-pro')}"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Lower values are more focused, higher values more creative."
    )
    
    # API Key status
    if "API_KEY" in credentials and credentials["API_KEY"]:
        st.success("API Key configured ✓")
    else:
        st.error("API Key not found in credentials.yaml")
    
    # File upload for RAG
    st.header("Document Upload for RAG")
    file_uploads = st.file_uploader(
        "Upload documents for the chatbot to reference",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx", "html", "md"]
    )
    
    if file_uploads:
        st.session_state.uploaded_files = file_uploads
        # Process button
        if st.button("Process Documents"):
            with st.spinner("Processing documents for RAG..."):
                st.session_state.vector_store = process_files(file_uploads)
                if st.session_state.vector_store:
                    st.session_state.file_processed = True
                    set_vector_store(st.session_state.vector_store)
                    st.success(f"Successfully processed {len(file_uploads)} document(s)")
                else:
                    st.error("Failed to process documents")
    
    # Document status
    if st.session_state.file_processed and st.session_state.vector_store:
        st.success("Documents processed and ready for RAG ✓")
        if st.button("Clear Documents"):
            st.session_state.vector_store = None
            set_vector_store(None)
            st.session_state.file_processed = False
            st.session_state.uploaded_files = []
            st.rerun()

# Main chat interface
st.title("Gemini AI Chatbot with LangChain RAG")

# Show RAG status
if st.session_state.file_processed and st.session_state.vector_store:
    st.success("🔍 RAG enabled: Responses will be enhanced with knowledge from your documents")
    # Display processed files info
    if st.session_state.uploaded_files:
        with st.expander("Uploaded Documents"):
            for file in st.session_state.uploaded_files:
                st.write(f"- {file.name}")
else:
    st.info("💡 Upload documents in the sidebar to enable Retrieval Augmented Generation (RAG)")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Format the chat history for LangChain
if st.session_state.messages:
    chat_history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            chat_history.append(
                (st.session_state.messages[i]["content"], 
                 st.session_state.messages[i+1]["content"])
            )
    st.session_state.chat_history = chat_history

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get model name from credentials
            model_name = load_credentials().get("AGENT_MODEL_NAME", "gemini-1.5-pro")
            
            # Generate response with sources
            response_text, sources = generate_response(
                st.session_state.messages,
                model_name,
                temperature
            )
            
            # Display response
            st.markdown(response_text)
            
            # Display source files if any were used
            if sources:
                st.divider()
                st.info(f"📄 **Files used for this response:**\n{', '.join(sources)}")
    
    # Store source information in session state if needed for future reference
    if not "sources" in st.session_state:
        st.session_state.sources = {}
    response_id = len(st.session_state.messages) // 2  # Unique ID for this Q&A pair
    st.session_state.sources[response_id] = sources
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Control buttons
col1, col2 = st.columns(2)

with col1:
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

with col2:
    # Reset all button (clears chat and documents)
    if st.button("Reset All"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        set_vector_store(None)
        st.session_state.file_processed = False
        st.session_state.uploaded_files = []
        st.rerun()
