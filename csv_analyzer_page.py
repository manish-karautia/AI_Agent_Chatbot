import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from io import StringIO

# Configure the Gemini API key from your .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper Function to Generate and Execute Code ---

def get_code_generation_prompt(df_head, user_query):
    """Creates a detailed prompt for the LLM to generate Python code."""
    
    # Get the column names from the dataframe head
    columns = df_head.columns.tolist()
    
    return f"""
    You are an expert Python data analyst. Your task is to write Python code to answer a user's question based on a given pandas DataFrame.

    **Instructions:**
    1.  The user's DataFrame is named `df`.
    2.  The first few rows of the DataFrame are:
        ```
        {df_head.to_string()}
        ```
    3.  The DataFrame has the following columns: `{columns}`.
    4.  The user's question is: "{user_query}"

    **Your Code's Task:**
    -   Write a Python script using the `df` DataFrame.
    -   **If the user asks for a plot or chart:**
        -   Use the `matplotlib.pyplot` library.
        -   Generate a single, clear, and well-labeled chart.
        -   **Crucially, you MUST save the plot to a file named 'output_chart.png'.** Do not display it with `plt.show()`.
    -   **If the user asks for data or a calculation:**
        -   Calculate the result.
        -   Store the final result (which could be a string, number, or a new DataFrame) in a variable named `result`.
    -   Do not include any sample data creation (e.g., `pd.DataFrame(...)`). The `df` is already provided.
    -   Your response must be ONLY the Python code, enclosed in ```python ... ```.

    **Example for plotting:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    df['Country'].value_counts().plot(kind='bar')
    plt.title('Sales by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output_chart.png')
    ```
    
    **Example for data retrieval:**
    ```python
    result = df[df['Sales'] > 500]
    ```
    """

def execute_generated_code(code, df):
    """Executes the generated Python code and returns the result."""
    
    # Create a safe environment to execute the code
    local_vars = {"df": df, "pd": pd}
    global_vars = {"plt": __import__("matplotlib.pyplot")}
    
    # Clean the code by removing markdown fences
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-len("```")].strip()

    try:
        # Execute the code
        exec(code, global_vars, local_vars)
        
        # Check for results
        if os.path.exists("output_chart.png"):
            # If a chart was created, return its path
            chart_path = "output_chart.png"
            return {"type": "plot", "path": chart_path}
        
        if "result" in local_vars:
            # If a data result was created, return it
            return {"type": "data", "value": local_vars["result"]}
            
        return {"type": "text", "value": "Code executed, but no result was captured."}

    except Exception as e:
        return {"type": "error", "value": f"An error occurred while executing the code: {e}"}


# --- Main Streamlit Page Function ---

def csv_analyzer_page():
    """The main function to create the CSV Analyzer page."""
    st.header("📊 CSV Analysis Agent (Code Generator)")
    st.write("Upload a CSV file and ask questions. The AI will write and run code to get your answer.")
    st.markdown("---")

    if "df" not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully! Here's a preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        st.write("### Ask your data a question")

        if "csv_chat_history" not in st.session_state:
            st.session_state.csv_chat_history = []

        for msg in st.session_state.csv_chat_history:
            with st.chat_message(msg["role"]):
                if msg.get("is_code"):
                    st.code(msg["content"], language="python")
                elif msg.get("is_plot"):
                    st.image(msg["content"])
                elif isinstance(msg["content"], pd.DataFrame):
                    st.dataframe(msg["content"])
                else:
                    st.write(msg["content"])

        if prompt := st.chat_input("e.g., Plot a bar chart of total sales by country"):
            st.session_state.csv_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking and writing code..."):
                    # Step 1: Generate the code
                    full_prompt = get_code_generation_prompt(st.session_state.df.head(), prompt)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(full_prompt)
                    generated_code = response.text

                    st.write("I've written this code to answer your question:")
                    st.code(generated_code, language="python")
                    st.session_state.csv_chat_history.append({"role": "assistant", "content": generated_code, "is_code": True})

                with st.spinner("Running the code..."):
                    # Step 2: Execute the code
                    result = execute_generated_code(generated_code, st.session_state.df)

                    # Step 3: Display the result
                    if result["type"] == "plot":
                        st.image(result["path"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": result["path"], "is_plot": True})
                        os.remove(result["path"]) # Clean up the created file
                    elif result["type"] == "data":
                        st.write("Here is the result:")
                        st.dataframe(result["value"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": result["value"]})
                    elif result["type"] == "error":
                        st.error(result["value"])
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": str(result["value"])})
                    else:
                        st.write(result.get("value", "Done."))
                        st.session_state.csv_chat_history.append({"role": "assistant", "content": result.get("value", "Done.")})

    else:
        st.info("Please upload a CSV file to start the conversation.")
