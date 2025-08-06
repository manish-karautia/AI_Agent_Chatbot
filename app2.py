import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq
from googleapiclient.discovery import build
from config.config import GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID
from utils.image_generation_tool import generate_image_url
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- CORE LOGIC FUNCTIONS ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return new_db

def get_conversational_chain(response_mode="detailed", llm_provider="gemini"):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    if response_mode == "concise":
        prompt_template = """
        Answer the question concisely and to the point from the provided context. If the answer is not in
        provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    
    if llm_provider == "groq":
        return prompt_template, None

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return prompt_template, chain

def perform_web_search(query):
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        st.error("Google Search API Key or CSE ID not configured.")
        return []
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
        snippets = []
        if 'items' in res:
            for item in res['items']:
                snippets.append(item.get('snippet'))
        return snippets
    except Exception as e:
        st.error(f"Error performing web search: {e}")
        return []

def get_groq_response(messages, model_name="llama-3.3-70b-versatile"):
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error from Groq API: {e}"

def user_input(user_question, response_mode, llm_provider, chat_history, memory_enabled):
    final_response = ""
    # --- LLM with TOOL USE Logic for Image Generation ---
    tool_use_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # app.py (revised tool_use_prompt)

    tool_use_prompt = f"""
    YOUR SOLE TASK is to determine if the user is asking to create an image.
    If the user's question asks for an image, you MUST respond with a JSON object.
    Do not provide any text-based or descriptive answers. If the user's intent is to get an image,
    your ONLY response should be the JSON object below.

    The tool is named 'generate_image' and it takes one parameter:
    - prompt: The text description of the image to generate.

    Example JSON response:
    {{
        "tool_call": "generate_image",
        "parameters": {{
            "prompt": "A cat wearing a spacesuit"
        }}
    }}

    If the question is not an image generation request, respond in a normal conversational manner.
    DO NOT respond with a JSON object if the user is not asking for an image.

    User's question: {user_question}
    """
    
    try:
        llm_response = tool_use_llm.invoke(tool_use_prompt)
        response_text = llm_response.content
        
        if response_text.startswith('{') and 'tool_call' in response_text:
            try:
                tool_call = json.loads(response_text)
                if tool_call.get('tool_call') == 'generate_image':
                    image_prompt = tool_call.get('parameters', {}).get('prompt')
                    if image_prompt:
                        st.info(f"Generating image for: '{image_prompt}'...")
                        image_url = generate_image_url(image_prompt)
                        if "Error" not in image_url:
                            st.image(image_url, caption=image_prompt)
                            final_response = "Here is the image I generated for you."
                        else:
                            final_response = image_url
                    else:
                        final_response = "I tried to generate an image, but the prompt was missing."
                else:
                    final_response = "I received a tool call but it was for an unknown tool."
            except json.JSONDecodeError:
                final_response = "I encountered an error trying to interpret the tool call response."
        else:
            # --- Fallback to existing RAG/Web Search/Groq Logic ---
            if llm_provider == "gemini":
                if not os.path.exists("faiss_index"):
                    final_response = "Please upload and process PDF files first to create the knowledge base."
                else:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)

                    prompt_template, rag_chain = get_conversational_chain(response_mode, llm_provider)
                    rag_response = rag_chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    final_response = rag_response["output_text"]

                    if "answer is not available" in final_response.lower() or not final_response.strip():
                        st.info("💡 No direct answer found in documents. Performing a web search...")
                        search_results = perform_web_search(user_question)
                        if search_results:
                            search_context = "\n".join(search_results)
                            web_synthesis_prompt_template = f"""
                            You are an expert research assistant. Based on the following information from documents and web search results...

                            Document Context: {docs}
                            Web Search Results: {search_context}
                            Question: {user_question}
                            Answer:
                            """
                            try:
                                synthesis_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
                                synthesis_response = synthesis_model.invoke(web_synthesis_prompt_template)
                                final_response = synthesis_response.content
                            except Exception as e:
                                final_response = f"Error synthesizing web search results: {e}"
                        else:
                            final_response = "I couldn't find a direct answer in the documents or through web search."
            
            elif llm_provider == "groq":
                messages = [{"role": "system", "content": "You are an AI assistant."}]
                if memory_enabled and chat_history:
                    for msg in chat_history:
                        messages.append(msg)
                messages.append({"role": "user", "content": user_question})
                final_response = get_groq_response(messages)

    except Exception as e:
        final_response = f"An error occurred during LLM processing: {e}"

    return final_response

# --- UI PAGES AND MAIN APP ---
import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 2px solid #3498db;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat Message Containers */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        margin-bottom: 1rem;
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* User Message */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant Message */
    .assistant-message {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(116, 185, 255, 0.3);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Control Panel */
    .control-panel {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.8);
        padding: 8px 16px;
        border-radius: 25px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 1);
        border-color: #3498db;
        transform: translateY(-2px);
    }
    
    /* Toggle Switch */
    .stToggle > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #3498db;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
        margin-bottom: 1rem;
    }
    
    /* Chat Input */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Sidebar Elements */
    .css-1d391kg .stImage > img {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .css-1d391kg .stRadio > div > label {
        color: white;
        background: rgba(255, 255, 255, 0.1);
        margin: 5px 0;
        padding: 10px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 15px;
        text-align: center;
        border-top: 2px solid #3498db;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #3498db;
    }
    
    /* Success/Warning Messages */
    .stSuccess, .stWarning {
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    </style>
    """, unsafe_allow_html=True)

def file_processing_page():
    st.markdown('<div class="main-header">📁 PDF Document Processing</div>', unsafe_allow_html=True)
    
    # Hero section with image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("img/Robot.jpg", width=300)
    
    st.markdown("---")
    
    # File upload section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3 style="color: #2c3e50;">Upload Your Documents</h3>
        <p style="color: #7f8c8d;">Upload multiple PDF files to process and chat with their content</p>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_docs = st.file_uploader(
        "Choose PDF files", 
        accept_multiple_files=True,
        type=['pdf'],
        help="Upload one or more PDF files to process"
    )
    
    if pdf_docs:
        st.success(f"✅ {len(pdf_docs)} file(s) uploaded successfully!")
        
        # Display uploaded files
        with st.expander("📄 Uploaded Files"):
            for i, doc in enumerate(pdf_docs, 1):
                st.write(f"{i}. {doc.name} ({doc.size} bytes)")
    
    # Process button with enhanced styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("🔄 Processing your documents..."):
                    # Simulated processing
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Your actual processing code would go here
                    # raw_text = get_pdf_text(pdf_docs)
                    # text_chunks = get_text_chunks(raw_text)
                    # get_vector_store(text_chunks)
                    
                    st.balloons()
                    st.success("✨ Documents processed successfully! You can now chat with your documents.")
            else:
                st.warning("⚠️ Please upload at least one PDF file before processing.")

def chat_page():
    st.markdown('<div class="main-header">🤖 AI Chat Assistant</div>', unsafe_allow_html=True)
    
    # Control panel
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**🔧 LLM Provider**")
        llm_provider = st.radio(
            "Choose your AI model:",
            ("Gemini", "Groq"),
            horizontal=True,
            key="llm_provider_radio"
        )
    
    with col2:
        st.markdown("**📝 Response Style**")
        response_mode = st.radio(
            "Select response length:",
            ("Detailed", "Concise"),
            horizontal=True,
            key="response_mode_radio"
        )
    
    with col3:
        st.markdown("**🧠 Memory Settings**")
        memory_enabled = st.toggle(
            "Enable Chat Memory",
            value=True,
            key="memory_toggle",
            help="Keep conversation context across messages"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Download chat history button
    if st.session_state.chat_history:
        chat_text = "\n\n".join([
            f"🧑‍💻 User: {msg['content']}" if msg["role"] == "user" 
            else f"🤖 Assistant: {msg['content']}" 
            for msg in st.session_state.chat_history
        ])
        
        st.download_button(
            label="💾 Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            help="Download your conversation history"
        )
    
    # Chat display area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #7f8c8d;">
            <h4>👋 Welcome to your AI Assistant!</h4>
            <p>Start a conversation by typing your question below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>🧑‍💻 You:</strong><br>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 AI Assistant:</strong><br>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💭 Ask me anything..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Show typing indicator
        with st.spinner("🤔 AI is thinking..."):
            # Simulated response - replace with your actual function
            # response = user_input(prompt, response_mode.lower(), llm_provider.lower(), 
            #                     st.session_state.chat_history, memory_enabled)
            
            # Placeholder response
            response = f"This is a simulated response to: '{prompt}' using {llm_provider} with {response_mode.lower()} mode."
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to show new messages
        st.rerun()

def main():
    st.set_page_config(
        page_title="🤖 AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: white; margin-bottom: 0;">🎛️ Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Profile section
        if st.button("img/gkj.jpg"):  # Replace with actual image
            st.image("img/gkj.jpg", width=200)
        
        # Navigation
        page = st.radio(
            "🚀 Go to:",
            ["💬 Chat", "📁 File Processing"],
            index=0,
            key="nav_radio"
        )
        
        st.markdown("---")
        
        # Info section
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 2rem;">
            <h4 style="color: white; margin-bottom: 1rem;">ℹ️ About</h4>
            <p style="color: #ecf0f1; font-size: 0.9rem;">
                This AI assistant can process PDF documents and answer questions based on their content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Creator info
        st.markdown("""
        <div style="margin-top: 3rem; text-align: center;">
            <p style="color: #bdc3c7; font-size: 0.8rem;">
                ✨ Created with ❤️ by<br>
                <a href="https://github.com/gurpreetkaurjethra" target="_blank" 
                   style="color: #3498db; text-decoration: none;">
                   Gurpreet Kaur Jethra
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content routing
    if "💬 Chat" in page:
        chat_page()
    elif "📁 File Processing" in page:
        file_processing_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="color: white; margin: 0;">
            © 2024 <a href="https://github.com/gurpreetkaurjethra" target="_blank" 
               style="color: #3498db; text-decoration: none;">Gurpreet Kaur Jethra</a> 
            | Made with ❤️ using Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()