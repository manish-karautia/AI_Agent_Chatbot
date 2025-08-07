import streamlit as st
import nest_asyncio
nest_asyncio.apply()

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq
from googleapiclient.discovery import build
from datetime import datetime
import time
import pandas as pd

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# API keys from environment
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- CORE HELPER FUNCTIONS (PDF & GENERAL) ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(response_mode="detailed"):
    detailed_template = """
    You are an expert research assistant. Your goal is to provide a detailed, easy-to-understand answer using ONLY the provided context.
    1. First, directly answer the user's question.
    2. Then, elaborate on the answer with all relevant details, facts, and figures found in the text.
    3. If the context includes steps, lists, or multiple key points, present them clearly using bullet points.
    4. Do not use any information outside of the provided context. If the answer is not in the context, you MUST say "I'm sorry, but the answer is not available in the provided documents."

    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    concise_template = """
    Based ONLY on the provided context, give a concise, one-sentence answer to the question.
    If the answer is not available in the context, state that clearly.

    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    prompt_template = detailed_template if response_mode == "detailed" else concise_template
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def perform_web_search(query):
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        st.error("Google Search API Key or CSE ID not configured.")
        return []
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
        return [item.get('snippet') for item in res.get('items', [])]
    except Exception as e:
        st.error(f"Error performing web search: {e}")
        return []

# --- CSV ANALYZER HELPER FUNCTIONS ---

def get_code_generation_prompt(df_head, user_query):
    columns = df_head.columns.tolist()
    return f"""
    You are an expert Python data analyst. Your task is to write Python code to answer a user's question based on a given pandas DataFrame.
    The user's DataFrame is named `df`. The first few rows are:
    ```
    {df_head.to_string()}
    ```
    The columns are: `{columns}`. The user's question is: "{user_query}"
    Your code must either save a plot to 'output_chart.png' using matplotlib or store a data result in a variable named `result`.
    Your response must be ONLY the Python code, enclosed in ```python ... ```.
    """

def execute_generated_code(code, df):
    local_vars = {"df": df, "pd": pd}
    global_vars = {"plt": __import__("matplotlib.pyplot")}
    
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-len("```")].strip()

    try:
        exec(code, global_vars, local_vars)
        if os.path.exists("output_chart.png"):
            return {"type": "plot", "path": "output_chart.png"}
        if "result" in local_vars:
            return {"type": "data", "value": local_vars["result"]}
        return {"type": "text", "value": "Code executed, but no result was captured."}
    except Exception as e:
        return {"type": "error", "value": f"An error occurred: {e}"}

# --- RESPONSE GENERATION LOGIC ---

def get_gemini_rag_response(user_question, response_mode):
    if not os.path.exists("faiss_index"):
        return "Please upload and process PDF files first."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    if not docs:
        final_response = "I'm sorry, but the answer is not available in the provided documents."
    else:
        chain = get_conversational_chain(response_mode)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        final_response = response["output_text"]

    if "answer is not available" in final_response.lower() or not final_response.strip():
        st.info("💡 No answer in documents. Searching the web...")
        search_results = perform_web_search(user_question)
        if search_results:
            search_context = "\n".join(search_results)
            web_synthesis_prompt = f"Synthesize info from Document Context and Web Search Results to answer.\n\nContext: {docs}\n\nWeb Search: {search_context}\n\nQuestion: {user_question}\n\nAnswer:"
            try:
                synthesis_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
                final_response = synthesis_model.invoke(web_synthesis_prompt).content
            except Exception as e:
                final_response = f"Error during web synthesis: {e}"
        else:
            final_response = "No direct answer in documents or via web search."
    return final_response

def get_groq_chat_response(user_question, response_mode, chat_history):
    system_prompt = ("You are an expert AI assistant. Provide comprehensive answers." if response_mode == "detailed" else "You are a concise AI assistant. Provide direct answers.")
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(msg for msg in chat_history if msg['role'] in ['user', 'assistant'])
    messages.append({"role": "user", "content": user_question})

    try:
        response = groq_client.chat.completions.create(messages=messages, model="llama3-70b-8192")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error from Groq API: {e}"

# --- CALLBACK & UI HELPER FUNCTIONS ---

def clear_chat_history():
    active_page = st.session_state.get("page", "Home")
    if active_page == "Document Chat":
        st.session_state.document_chat_history = []
    elif active_page == "CSV Analysis":
        st.session_state.csv_chat_history = []
    else:
        st.session_state.chat_history = []

def regenerate_last_response():
    active_page = st.session_state.get("page", "Home")
    history_key_map = {
        "Document Chat": "document_chat_history",
        "General Chat": "chat_history"
    }
    history_key = history_key_map.get(active_page)
    if not history_key or len(st.session_state.get(history_key, [])) < 2:
        return

    last_user_question = st.session_state[history_key][-2]['content']
    new_response_mode = st.session_state.response_mode.lower()

    with st.spinner(f"Re-generating..."):
        if active_page == "Document Chat":
            new_response = get_gemini_rag_response(last_user_question, new_response_mode)
        else:
            history_for_regeneration = st.session_state[history_key][:-1]
            new_response = get_groq_chat_response(last_user_question, new_response_mode, history_for_regeneration)
        st.session_state[history_key][-1] = {"role": "assistant", "content": new_response}

# --- UI PAGES ---

def welcome_page():
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🧠 Welcome to Agentic AI</h1>
        <p class="hero-subtitle">Your Intelligent Assistant for Documents, Data, and Conversations</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📄</div><h3>Document Intelligence</h3><p>Upload PDFs and get intelligent answers powered by Gemini AI.</p></div>""", unsafe_allow_html=True)
        if st.button("🚀 Start Document Chat", use_container_width=True, type="primary"):
            st.session_state.page = "Document Chat"
            st.rerun()
    with col2:
        st.markdown("""<div class="feature-card"><div class="feature-icon">💬</div><h3>General Conversation</h3><p>Engage in dynamic conversations with our fast AI assistant powered by Groq.</p></div>""", unsafe_allow_html=True)
        if st.button("💭 Start General Chat", use_container_width=True, type="secondary"):
            st.session_state.page = "General Chat"
            st.rerun()
    with col3:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📊</div><h3>CSV Analysis</h3><p>Upload a CSV and generate charts and insights with natural language.</p></div>""", unsafe_allow_html=True)
        if st.button("📈 Start CSV Analysis", use_container_width=True, type="secondary"):
            st.session_state.page = "CSV Analysis"
            st.rerun()

def document_chat_page():
    st.markdown("""<div class="page-header"><h1>📄 Document Intelligence</h1><p>Upload your PDFs and unlock intelligent document analysis</p></div>""", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Process Documents", use_container_width=True, type="primary", disabled=not pdf_docs):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    if os.path.exists("faiss_index"):
        st.markdown("<h3>💬 Chat with Your Documents</h3>", unsafe_allow_html=True)
        if "document_chat_history" not in st.session_state:
            st.session_state.document_chat_history = []
        
        st.radio("Response Style:", ("Detailed", "Concise"), key="response_mode", on_change=regenerate_last_response, horizontal=True)

        for msg in st.session_state.document_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.document_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = get_gemini_rag_response(prompt, st.session_state.response_mode.lower())
                    st.markdown(response)
                    st.session_state.document_chat_history.append({"role": "assistant", "content": response})
    else:
        st.info("Upload and process PDF documents to start chatting.")

def chat_page():
    st.markdown("""<div class="page-header"><h1>💬 General Conversation</h1><p>Chat with our lightning-fast AI assistant</p></div>""", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.radio("Response Style:", ("Detailed", "Concise"), key="response_mode", on_change=regenerate_last_response, horizontal=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_groq_chat_response(prompt, st.session_state.response_mode.lower(), st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

def csv_analyzer_page():
    st.markdown("""<div class="page-header"><h1>📊 CSV Analysis Agent</h1><p>Upload a CSV file and ask questions to generate insights and charts.</p></div>""", unsafe_allow_html=True)
    if "df" not in st.session_state:
        st.session_state.df = None
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully! Here's a preview:")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.df = None
    if st.session_state.df is not None:
        st.write("### Ask your data a question")
        if "csv_chat_history" not in st.session_state:
            st.session_state.csv_chat_history = []
        for msg in st.session_state.csv_chat_history:
            with st.chat_message(msg["role"]):
                if msg.get("is_code"): st.code(msg["content"], language="python")
                elif msg.get("is_plot"): st.image(msg["content"])
                elif isinstance(msg["content"], pd.DataFrame): st.dataframe(msg["content"])
                else: st.write(msg["content"])
        if prompt := st.chat_input("e.g., Plot a bar chart of total sales by country"):
            st.session_state.csv_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking and writing code..."):
                    full_prompt = get_code_generation_prompt(st.session_state.df.head(), prompt)
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    response = model.generate_content(full_prompt)
                    generated_code = response.text
                    st.write("I've written this code to answer your question:")
                    st.code(generated_code, language="python")
                    st.session_state.csv_chat_history.append({"role": "assistant", "content": generated_code, "is_code": True})
                with st.spinner("Running the code..."):
                    result = execute_generated_code(generated_code, st.session_state.df)
                    if result["type"] == "plot":
                        st.image(result["path"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": result["path"], "is_plot": True})
                        os.remove(result["path"])
                    elif result["type"] == "data":
                        st.write("Here is the result:")
                        st.dataframe(result["value"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": result["value"]})
                    else:
                        st.error(result["value"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": str(result["value"])})
    else:
        st.info("Please upload a CSV file to start the conversation.")

def main():
    st.set_page_config(page_title="Agentic AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --primary-color: #667eea; --secondary-color: #764ba2; --background-dark: #0f0f23;
        --background-light: #1a1a2e; --surface-color: #16213e; --text-primary: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, var(--background-light) 100%);
        font-family: 'Inter', sans-serif; color: var(--text-primary);
    }
    #MainMenu, footer, header { visibility: hidden; }
    .css-1d391kg { background: var(--surface-color); }
    .hero-section { text-align: center; padding: 3rem 0; background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); border-radius: 20px; margin: 2rem 0; }
    .hero-title { font-size: 3.5rem; font-weight: 700; }
    .page-header h1 { font-size: 2.5rem; background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .feature-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; margin: 1rem 0; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important; color: white !important; }
    .stButton > button[kind="secondary"] { background: rgba(255,255,255,0.08) !important; color: var(--text-primary) !important; }
    </style>""", unsafe_allow_html=True)

    if "page" not in st.session_state: st.session_state.page = "Home"
    if "response_mode" not in st.session_state: st.session_state.response_mode = "Detailed"

    with st.sidebar:
        st.markdown("""<div style="text-align: center; padding: 1rem 0;"><h1 style="font-size: 2rem; margin: 0; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🧠 Agentic AI</h1></div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("✨ New Session", use_container_width=True, type="primary"):
            clear_chat_history()
            st.success("🎉 Session cleared!")
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        if st.button("🏠 Home", use_container_width=True, type="secondary"): st.session_state.page = "Home"; st.rerun()
        if st.button("💬 General Chat", use_container_width=True, type="secondary"): st.session_state.page = "General Chat"; st.rerun()
        if st.button("📄 Document Chat", use_container_width=True, type="secondary"): st.session_state.page = "Document Chat"; st.rerun()
        if st.button("📊 CSV Analysis", use_container_width=True, type="secondary"): st.session_state.page = "CSV Analysis"; st.rerun()

    page_map = {
        "Home": welcome_page,
        "General Chat": chat_page,
        "Document Chat": document_chat_page,
        "CSV Analysis": csv_analyzer_page
    }
    page_map[st.session_state.page]()

if __name__ == "__main__":
    main()
