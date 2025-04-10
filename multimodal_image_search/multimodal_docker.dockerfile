# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY multimodal_search_streamlit.py .

# Create directories for caches
RUN mkdir -p /root/.cache/torch
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /app/chromadb

# Set environment variables
ENV TORCH_HOME=/root/.cache/torch
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "multimodal_search_streamlit.py"]
