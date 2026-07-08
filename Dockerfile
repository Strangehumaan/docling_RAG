# Use Python 3.11 slim image
FROM python:3.11-slim-bookworm

# Set environment variables for clean logs and paths
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://ollama:11434 \
    CHROMA_DB_PATH=/app/chroma_db

WORKDIR /app

# Install system dependencies (curl for healthchecks, libgl1/libglib for OpenCV/Docling/RapidOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck to verify Streamlit container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
