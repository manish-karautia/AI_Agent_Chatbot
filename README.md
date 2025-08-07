🧠 Agent_AI: The All-in-One Intelligent Assistant
Welcome to Agent_AI! 🚀 This is a powerful, multi-functional Streamlit application that combines document analysis, data visualization, and general conversation into a single, easy-to-use interface. Powered by Google Gemini and Groq, Agent_AI is your go-to tool for intelligent interaction with your data and documents.

🎯 Key Features
📄 Document Intelligence (RAG):

Upload one or more PDF documents.

Ask complex questions about the content. The AI uses Retrieval-Augmented Generation (RAG) with a FAISS vector store to find precise answers.

Live Web Search Integration: If the answer isn't in the documents, the agent automatically performs a Google search to find the information online and synthesizes a complete answer.

📊 CSV Analysis & Visualization:

Upload a CSV file.

Ask the AI to perform data analysis or generate charts in plain English (e.g., "plot a bar chart of sales by country").

The AI writes and executes Python code on the fly to generate visualizations and data tables.

💬 High-Speed General Chat:

Engage in fast, dynamic conversations on any topic, powered by the high-speed Groq inference engine running LLaMA 3.

⚙️ Dynamic Response Control:

Easily switch between "Detailed" and "Concise" response styles. The AI will instantly regenerate its last answer to match your preferred level of detail.

🛠️ How It Works
Agent_AI integrates several cutting-edge technologies to deliver its features:

Document Chat (RAG): When you upload PDFs, the text is extracted, split into chunks, and converted into numerical representations (embeddings) using Google's models. These are stored in a FAISS vector database. When you ask a question, the system finds the most relevant text chunks and feeds them to Google Gemini as context to generate a precise answer.

CSV Analysis: You provide a CSV and a prompt. The app sends the column headers, a sample of the data, and your request to Google Gemini, instructing it to write matplotlib code. This code is then executed in the backend, and the resulting chart or data table is displayed in the chat.

General Chat: Your questions are sent directly to the Groq API, which provides near-instant responses from the LLaMA 3 language model.

🔧 Tech Stack & Requirements
This project is built with Python and relies on the following key libraries and services:

Frontend: Streamlit

LLM Orchestration: LangChain

LLMs: Google Gemini & Groq (LLaMA 3)

Embeddings & Vector Store: Google Generative AI Embeddings & FAISS

Core Libraries: PyPDF2, Pandas, google-api-python-client

🔑 API Keys Setup
To run this application, you need to get API keys from a few services.

Create a file named .env in the main project directory (Agentic_AI/).

Add the following keys to the file, replacing <YOUR_KEY_HERE> with your actual keys:

# Get from Google AI Studio -> https://ai.google.dev/
GOOGLE_API_KEY="<YOUR_KEY_HERE>"

# Get from Google Cloud Console (for Custom Search API) -> https://console.cloud.google.com/
GOOGLE_SEARCH_API_KEY="<YOUR_KEY_HERE>"
GOOGLE_CSE_ID="<YOUR_CUSTOM_SEARCH_ENGINE_ID>"

# Get from GroqCloud -> https://console.groq.com/keys
GROQ_API_KEY="<YOUR_KEY_HERE>"

▶️ Installation & How to Run
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
Open your terminal and clone the project files:

git clone https://github.com/your-username/Agent_AI.git
cd Agent_AI

2. Create a Virtual Environment
It is highly recommended to use a virtual environment. This project is stable on Python 3.11.

# Create the environment
python -m venv .venv

# Activate it (on Windows PowerShell)
.\.venv\Scripts\Activate.ps1

3. Install Dependencies
Install all the required libraries using the requirements.txt file:

pip install -r requirements.txt

4. Add Your API Keys
Create the .env file as described in the "API Keys Setup" section above and add your keys.

5. Run the Application
You're all set! Run the following command in your terminal:

streamlit run app.py

The application will open automatically in your default web browser.

💡 Usage
Navigate: Use the sidebar to switch between "Home", "General Chat", "Document Chat", and "CSV Analysis".

Document Chat:

Go to the "Document Chat" page.

Upload one or more PDF files.

Click the "Process Documents" button and wait for it to finish.

Ask questions in the chat box at the bottom.

CSV Analysis:

Go to the "CSV Analysis" page.

Upload a CSV file.

Ask for analysis or charts (e.g., "Show me a histogram of the age column").

General Chat:

Go to the "General Chat" page and start asking questions.LLM Project do drop ⭐ to this repo**
#### Follow me on [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gurpreetkaurjethra/) &nbsp; [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GURPREETKAURJETHRA/)

---
