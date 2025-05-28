# Agentic RAG Chatbot with Gemini AI and Streamlit

This demo project provides a baseline for implementing an agentic approach to Retrieval Augmented Generation (RAG) using Gemini AI and Streamlit. It demonstrates how to create an interactive chatbot with document understanding capabilities that can reason over user-provided documents.

## Features

- Modern chat interface built with Streamlit's `st.chat_message`.
- Integration with Google's Gemini AI through both direct API and LangChain.
- RAG capabilities using LangChain's document processing and retrieval.
- Support for multiple document types (pdf, docx, txt, csv, xlsx, html, md).
- Document upload and processing interface in the sidebar.
- Session management for chat history and document persistence.
- Temperature control for response creativity.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ccomkhj/llm-chatbot-demo.git
    cd llm-chatbot-demo
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Google API Setup:**

    * Get a Google API key for Gemini AI:
      1. Go to the [Google AI Studio](https://makersuite.google.com/app/apikey)
      2. Create an API key
      3. Create a `credentials.yaml` file in the project root with:
         ```yaml
         API_KEY: "your_gemini_api_key_here"
         AGENT_MODEL_NAME: "gemini-2.5-flash-preview-04-17"  # or another Gemini model
         ```

## Running the Application

1.  **Activate your virtual environment** (if not already active).

2.  **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

3.  **Open your browser** to the URL provided by Streamlit (usually `http://localhost:8501`).

4.  **Upload Documents (Optional):**
    * In the sidebar, use the document upload section to add files you want the chatbot to reference.
    * Click the "Process Documents" button to extract and index the content.
    * The app will display a success message when documents are ready to be used.

5.  **Chat with the Model:**
    * Type your questions in the chat input at the bottom of the page.
    * If documents were processed, the model will incorporate relevant information from them.
    * Sources used from your uploaded documents will be displayed below each response.
    * Use the "Clear Chat" button to start a new conversation or "Reset All" to clear both chat history and documents.

## Understanding the Agentic Approach to RAG

This demo implements a baseline for an agentic approach to RAG, which combines:

1. **Document Understanding**: The ability to process and index various document types.
2. **Semantic Retrieval**: Finding relevant document chunks based on query similarity.
3. **Context Integration**: Incorporating retrieved information into model responses.
4. **Conversation Memory**: Maintaining chat history to provide coherent responses across turns.

The implementation uses:

- **LangChain** for document processing, vectorization, and retrieval
- **Google's Generative AI** for powerful language model capabilities
- **Streamlit** for an interactive user interface

## Architecture

The system consists of three main components:

1. **Document Processor**: Handles file uploads, text extraction, and vector store creation
2. **LLM Interface**: Manages communication with Gemini AI models through direct API or LangChain
3. **Chat Interface**: Provides the user-facing components for interaction

The agentic nature comes from the system's ability to:

- Decide when to retrieve information from documents
- Select the most relevant context pieces
- Synthesize information with the user's query
- Maintain conversational context across multiple turns

## Notes

* Document processing is handled in memory for simplicity. For production use, consider a persistent vector database.
* The implementation automatically switches between RAG and direct model access based on whether documents are uploaded.
* Temperature control allows adjusting the balance between deterministic and creative responses.
* For more advanced agentic capabilities, the system could be extended with:

  - Tool use for external API calls
  - Multi-step reasoning chains
  - Self-reflection mechanisms
  - Structured output parsing
