# Docling Local RAG - Layout-Aware Document Q&A

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system powered by **IBM's Docling** for advanced, layout-aware document parsing, **ChromaDB** for vector storage, and **Ollama** for local LLM inference.

This repository demonstrates how to build a high-fidelity document assistant that doesn't just read plain text, but understands document hierarchy, parses complex tables, and extracts diagrams/figures to display them directly alongside grounding references in chat.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b166ea86-c521-4dec-91b4-30b2523600f8" alt="App Dashboard Preview" width="100%" />
</p>

---

## Key Features

* **Layout-Aware Parsing (IBM Docling)**: Converts PDFs to structural markdown, preserving document hierarchy (headings, sections), reading order, and complex tables (serialized to Markdown tables).
* **Figure & Image Grounding**: Automatically extracts pictures and diagrams from documents and matches them to retrieved sections during queries.
* **100% Offline & Local**: Runs completely on your own machine using Ollama (`qwen2.5:7b-instruct` and `nomic-embed-text`).
* **Interactive UI**: Includes a Streamlit interface featuring:
  * **Reference Chat**: Query your document library with direct section/page citations and extracted figures.
  * **Document Ingestion**: Upload, parse, chunk, and embed new documents into the database.
* **Docker Support**: Pre-configured services with optional NVIDIA GPU hardware acceleration.

---

## Technology Stack

* **Parser**: [IBM Docling](https://github.com/DS4SD/docling)
* **LLM & Embeddings**: [Ollama](https://ollama.com/) (Qwen 2.5 & Nomic Embed)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Frontend**: [Streamlit](https://streamlit.io/)
* **Containers**: Docker & Docker Compose

---

## Repository Structure

```text
├── app.py                      # Streamlit entrypoint: Technical Reference Chat UI
├── rag_core.py                 # Core RAG pipeline (Docling parsing, embedding, querying)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Builds the Streamlit app container
├── docker-compose.yml          # Service definition (Ollama & Web App)
├── UI/                         # UI components (sidebar layout, connections)
├── pages/                      # Streamlit sub-pages (document ingestion UI)
├── materials/                  # UI assets and logos
├── manual/                     # Folder for storing raw PDF manuals
└── extracted_images/           # Automatically extracted figures and diagrams
```

---

## Quick Start (Local Run)

### Prerequisites
* Python 3.11 installed.
* [Ollama](https://ollama.com/) installed and running.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/docling-local-rag.git
   cd docling-local-rag
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Pull the required models in Ollama:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull nomic-embed-text
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## Quick Start (Docker Run)

You can run the application containerized in two ways: pulling the pre-built image from Docker Hub (quickest) or building it locally.

### Option A: Pull Pre-built Image from Docker Hub (Zero Setup)
You do not need to download the source repository. Simply save the following config as `docker-compose.yml` in an empty folder:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_storage:/root/.ollama
    restart: unless-stopped

  web:
    image: strangehumaan/cn-rag-web:latest
    container_name: siemens-rag-web
    ports:
      - "8501:8501"
    volumes:
      - ./manual:/app/manual
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - CHROMA_DB_PATH=/app/chroma_db
      - DEFAULT_MANUAL_PATH=/app/manual/AS410-5H_System_Manuals7400_cpu_410_proc_autom_smart_system_system_en-US_en-US.pdf
    depends_on:
      - ollama
    restart: unless-stopped

volumes:
  ollama_storage:
```

Then run:
```bash
docker-compose up -d
```

### Option B: Build Locally from Source
If you have cloned the repository, you can build the image locally and start the services:
```bash
docker-compose up --build -d
```

Once running, navigate to `http://localhost:8501` in your browser. The application will automatically check for and download the required Ollama models in the sidebar background.

*To enable NVIDIA GPU acceleration for Ollama, uncomment the `deploy` block under the `ollama` service in `docker-compose.yml`.*

---

## How it Works (Under the Hood)

### 1. Document Parsing & Image Extraction
When a PDF is uploaded, **Docling** parses the document structure. It extracts text sections, markdown tables, and crops out figures/drawings. The coordinates of these elements are stored, and pictures are saved to the `extracted_images/` folder.

### 2. Hierarchical Chunking
Instead of splitting text using arbitrary character limits, the system uses Docling's **Hierarchical Chunker** to group text semantically under its corresponding parent headings. Large chunks are then split with overlapping sliding windows.

### 3. Vector Embedding & Retrieval
Each chunk is embedded using `nomic-embed-text` with a prefix task (`search_document: `) and stored in **ChromaDB**. When a query is made, it is embedded (`search_query: `) and a cosine similarity search retrieves the top relevant context blocks.

### 4. Grounded Synthesis
The retrieved sections, table markdown, and any extracted page figures are sent to `qwen2.5:7b-instruct` inside a safety-oriented system prompt. The model synthesizes the answer, citing the exact page and section, and displays the related diagrams in the chat.
