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
        # Groq doesn't use langchain's load_qa_chain directly for this example.
        # We'll handle its call directly in user_input for simplicity.
        return prompt_template, None # Return the prompt template and a placeholder

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
    if llm_provider == "gemini":
        if not os.path.exists("faiss_index"):
            return "Please upload and process PDF files first to create the knowledge base."

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
        return final_response
    
    elif llm_provider == "groq":
        messages = [{"role": "system", "content": "You are an AI assistant."}]
        if memory_enabled and chat_history:
            for msg in chat_history:
                messages.append(msg)
        messages.append({"role": "user", "content": user_question})
        
        return get_groq_response(messages)

# --- UI PAGES AND MAIN APP ---

def file_processing_page():
    st.header("📁 PDF File's Section")
    st.image("img/Robot.jpg")
    st.write("---")
    pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done! You can now go to the Chat page.")
        else:
            st.warning("Please upload at least one PDF file.")

def chat_page():
    st.header("🤖 AI Chat Agent")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        llm_provider = st.radio("Select LLM Provider:", ("Gemini", "Groq"), horizontal=True)
    with col2:
        response_mode = st.radio("Select Response Style:", ("Detailed", "Concise"), horizontal=True)
    with col3:
        memory_enabled = st.toggle("Enable Chat Memory", value=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.chat_history:
        chat_text = "\n\n".join(
            [f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}" for msg in st.session_state.chat_history]
        )
        st.download_button(
            label="💾 Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(
                    prompt, 
                    response_mode.lower(), 
                    llm_provider.lower(), 
                    st.session_state.chat_history, 
                    memory_enabled
                )
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(
        page_title="Multi-Provider Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for a more appealing UI
    st.markdown("""
    <style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .stHeader { background-color: #1a1a2e; }
    .stSidebar { background-color: #16213e; padding: 20px; border-right: 2px solid #0f3460; }
    .stButton>button { background-color: #e94560; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-size: 16px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .stButton>button:hover { background-color: #b83f52; transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); }
    .stTextInput>div>div>input { background-color: #0f3460; color: #e0e0e0; border-radius: 8px; border: 1px solid #e94560; padding: 10px; }
    .stFileUploader>div>div>div>button { background-color: #0f3460; color: #e0e0e0; border: 1px dashed #e94560; border-radius: 8px; padding: 15px; }
    .stFileUploader>div>div>div>button:hover { background-color: #16213e; }
    .stRadio > label { color: #e0e0e0; font-weight: bold; }
    .stRadio > div > label > div:first-child { border-color: #e94560; }
    .stRadio > div > label > div:first-child > div { background-color: #e94560; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e94560; }
    .stMarkdown a { color: #e94560; }
    .stChatInput > div > div > textarea { background-color: #0f3460; color: #e0e0e0; border-radius: 8px; border: 1px solid #e94560; padding: 10px; }
    .stChatInput > div > div > button { background-color: #e94560; color: white; border-radius: 8px; border: none; padding: 10px 20px; }
    .stSpinner > div > div { color: #e94560; }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Navigation")
        st.image("img/gkj.jpg")
        page = st.radio(
            "Go to:",
            ["Chat", "File Processing"],
            index=0
        )
        st.write("---")
        st.write("AI App created by @ GurpreetKaur")

    if page == "Chat":
        chat_page()
    elif page == "File Processing":
        file_processing_page()
    
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center; border-top: 1px solid #0f3460;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank" style="color: #e94560;">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()



