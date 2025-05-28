import google.generativeai as genai
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain

from utils import load_credentials

# Function to generate response using LangChain with RAG or direct genai API
def generate_response_with_rag(query, chat_history, model_name, temperature=0.7):
    # Get credentials
    creds = load_credentials()
    
    # Configure API key
    if "API_KEY" in creds and creds["API_KEY"]:
        api_key = creds["API_KEY"]
    else:
        return "Error: API Key not found in credentials.yaml"
    
    try:
        # If we have a vector store, use RAG approach with LangChain
        if hasattr(generate_response_with_rag, "vector_store") and generate_response_with_rag.vector_store:
            # Configure the Gemini API for embeddings
            genai.configure(api_key=api_key)
            
            # Configure LangChain's ChatVertexAI (using API key only)
            llm = ChatVertexAI(
                model_name=model_name,
                temperature=temperature,
                model_kwargs={"google_api_key": api_key}
            )
        
            # Create a retriever from the vector store
            retriever = generate_response_with_rag.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create a ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            # Generate response with RAG
            result = qa_chain({"question": query, "chat_history": chat_history})
            
            # Format response with sources
            answer = result["answer"]
            source_docs = result["source_documents"]
            
            # Collect source information if available
            sources = set()
            if source_docs:
                for doc in source_docs:
                    if "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
            
            # Return both the answer and the sources
            return answer, list(sources)
        else:
            # No RAG, use direct genai API instead of LangChain
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # Create the model
            model = genai.GenerativeModel(model_name)
            
            # Format the conversation history for Gemini
            formatted_chat = []
            
            # Add instruction as a user message instead of system (Gemini only supports user/model roles)
            formatted_chat.append({
                "role": "user",
                "parts": ["You are a helpful AI assistant. Please remember this in your responses."]
            })
            
            # Add AI acknowledgment
            formatted_chat.append({
                "role": "model",
                "parts": ["I understand. I'm a helpful AI assistant and will respond accordingly."]
            })
            
            # Add past conversation messages in the format Gemini expects
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    formatted_chat.append({
                        "role": "user",
                        "parts": [human_msg]
                    })
                    formatted_chat.append({
                        "role": "model",
                        "parts": [ai_msg]
                    })
            
            # Add the current question
            formatted_chat.append({
                "role": "user",
                "parts": [query]
            })
            
            # Generate response using the Gemini API
            response = model.generate_content(
                formatted_chat,
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 0,
                }
            )
            
            # No sources for non-RAG responses
            return response.text, []
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# Set vector_store for RAG
def set_vector_store(vector_store):
    generate_response_with_rag.vector_store = vector_store

# Legacy function for compatibility
def generate_response(messages, model_name, temperature=0.7):
    # Get credentials
    creds_legacy = load_credentials()
    
    # Configure API key
    if "API_KEY" in creds_legacy and creds_legacy["API_KEY"]:
        # API key is validated in generate_response_with_rag
        pass
    else:
        return "Error: API Key not found in credentials.yaml"
    
    try:
        # Convert streamlit messages to chat history format for LangChain
        chat_history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                chat_history.append((messages[i]["content"], messages[i+1]["content"]))
        
        # Get the last message (user query)
        last_message = messages[-1]["content"] if messages else ""
        
        # Use the RAG-based response generator
        return generate_response_with_rag(last_message, chat_history, model_name, temperature)
    except ValueError as e:
        return f"Value error in response generation: {str(e)}", []
    except KeyError as e:
        return f"Key error in response generation: {str(e)}", []
    except IOError as e:
        return f"I/O error in response generation: {str(e)}", []
    except RuntimeError as e:
        return f"Runtime error in response generation: {str(e)}", []
    except Exception as e:  # Still keep a generic exception handler as fallback
        return f"Error generating response: {str(e)}", []
