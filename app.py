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

# Import necessary modules for web search (assuming you've set them up as discussed)
from googleapiclient.discovery import build
from config.config import GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Adjusted chunk_size and chunk_overlap for better performance and context
    # 50000 is very large; typically chunks are much smaller (e.g., 1000-2000)
    # For academic papers, you might need larger chunks, but be mindful of LLM context window limits.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # IMPORTANT: Set allow_dangerous_deserialization=True as per previous error fix
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return new_db

def get_conversational_chain(response_mode="detailed"):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    # Adjust prompt based on response_mode
    if response_mode == "concise":
        prompt_template = """
        Answer the question concisely and to the point from the provided context. If the answer is not in
        provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def perform_web_search(query):
    """
    Performs a real-time web search using Google Custom Search API.
    Returns a list of snippets.
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        st.error("Google Search API Key or CSE ID not configured. Please check config/config.py and .env file.")
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

def user_input(user_question, response_mode):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDF files first to create the knowledge base.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # First, try to get the answer from the PDF context (RAG)
    rag_chain = get_conversational_chain(response_mode)
    rag_response = rag_chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    final_response = rag_response["output_text"]

    # Check if RAG couldn't find the answer and then perform web search
    if "answer is not available in the context" in final_response.lower() or not final_response.strip():
        st.info("💡 No direct answer found in documents. Performing a web search...")
        search_results = perform_web_search(user_question)
        
        if search_results:
            search_context = "\n".join(search_results)
            web_synthesis_prompt_template = f"""
            You are an expert research assistant. Based on the following information from documents and web search results,
            provide a detailed and helpful answer to the question. Prioritize information from documents if available and relevant.
            If the information is not sufficient from both sources, state that you cannot provide a complete answer.

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
                final_response = f"Error synthesizing web search results: {str(e)}"
        else:
            final_response = "I couldn't find a direct answer in the documents or through web search."

    st.write("Reply: ", final_response)


def main():
    # Custom CSS for a more appealing UI
    st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a2e; /* Dark background */
        color: #e0e0e0; /* Light text */
        font-family: 'Inter', sans-serif; /* Modern font */
    }
    .stHeader {
        background-color: #1a1a2e;
    }
    .stSidebar {
        background-color: #16213e; /* Slightly different dark background for sidebar */
        padding: 20px;
        border-right: 2px solid #0f3460;
    }
    .stButton>button {
        background-color: #e94560; /* Accent color for buttons */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #b83f52;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stTextInput>div>div>input {
        background-color: #0f3460; /* Darker input background */
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #e94560;
        padding: 10px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #0f3460;
        color: #e0e0e0;
        border: 1px dashed #e94560;
        border-radius: 8px;
        padding: 15px;
    }
    .stFileUploader>div>div>div>button:hover {
        background-color: #16213e;
    }
    .stRadio > label {
        color: #e0e0e0;
        font-weight: bold;
    }
    .stRadio > div > label > div:first-child {
        border-color: #e94560; /* Radio button border */
    }
    .stRadio > div > label > div:first-child > div {
        background-color: #e94560; /* Radio button fill */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e94560; /* Accent color for headers */
    }
    .stMarkdown a {
        color: #e94560; /* Accent color for links */
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #0f3460;
        text-align: right;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .chat-message.assistant {
        background-color: #16213e;
        text-align: left;
        border-bottom-left-radius: 0;
    }
    .stSpinner > div > div {
        color: #e94560; /* Spinner color */
    }
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:", layout="wide")
    
    # Use columns for better layout
    col1, col2 = st.columns([0.7, 0.3]) # Adjust column width for chat vs sidebar

    with col1: # Main content area
        st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

        # Add Response Mode selection
        response_mode = st.radio(
            "Select Response Style:",
            ("Detailed", "Concise"),
            index=0, # Default to Detailed
            horizontal=True,
            key="response_mode_radio"
        )
        st.markdown("---") # Separator

        user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝", placeholder="E.g., Summarize the key findings from the documents...")

        if user_question:
            user_input(user_question, response_mode.lower())

    with col2: # Sidebar content area
        st.sidebar.image("img/Robot.jpg")
        st.sidebar.write("---")
        
        st.sidebar.title("📁 PDF File's Section")
        pdf_docs = st.sidebar.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.sidebar.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."): # user friendly message.
                    raw_text = get_pdf_text(pdf_docs) # get the pdf text
                    text_chunks = get_text_chunks(raw_text) # get the text chunks
                    get_vector_store(text_chunks) # create vector store
                    st.sidebar.success("Done!")
            else:
                st.sidebar.warning("Please upload at least one PDF file.")
            
        st.sidebar.write("---")
        st.sidebar.image("img/gkj.jpg")
        st.sidebar.write("AI App created by @ GurpreetKaur")
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