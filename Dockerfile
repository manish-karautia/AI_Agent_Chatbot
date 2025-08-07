# Use an official lightweight Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app1

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Define the command to run your Streamlit app1
# This uses the default Streamlit port 8501
CMD ["streamlit", "run", "app1.py"]