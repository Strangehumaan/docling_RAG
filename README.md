# Docling Local RAG - Layout-Aware Document Q&A

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system powered by **IBM's Docling** for advanced, layout-aware document parsing, **ChromaDB** for vector storage, and **Ollama** for local LLM inference.

This repository demonstrates how to build a high-fidelity document assistant that doesn't just read plain text, but understands document hierarchy, parses complex tables, and extracts diagrams/figures to display them directly alongside grounding references in chat.

---

## 🚀 Key Features

* **Layout-Aware Parsing (IBM Docling)**: Converts PDFs to structural markdown, preserving document hierarchy (headings, sections), reading order, and complex tables (serialized to Markdown tables).
* **Figure & Image Grounding**: Automatically extracts pictures and diagrams from documents and matches them to retrieved sections during queries.
* **100% Offline & Local**: Runs completely on your own machine using Ollama (`qwen2.5:7b-instruct` and `nomic-embed-text`).
* **Interactive UI**: Includes a Streamlit interface featuring:
  * **Reference Chat**: Query your document library with direct section/page citations and extracted figures.
  * **Document Ingestion**: Upload, parse, chunk, and embed new documents into the database.
* **Docker Support**: Pre-configured services with optional NVIDIA GPU hardware acceleration.

---

## 🛠️ Technology Stack

* **Parser**: [IBM Docling](https://github.com/DS4SD/docling)
* **LLM & Embeddings**: [Ollama](https://ollama.com/) (Qwen 2.5 & Nomic Embed)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Frontend**: [Streamlit](https://streamlit.io/)
* **Containers**: Docker & Docker Compose

---

## 📁 Repository Structure

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

## 🚀 Quick Start (Local Run)

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

## 🐳 Quick Start (Docker Run)

To run the application using Docker:

1. Build and launch the containers:
   ```bash
   docker-compose up --build -d
   ```
2. Open your browser and navigate to `http://localhost:8501`.
3. The application will automatically check for and download the required Ollama models in the sidebar background.

*To enable NVIDIA GPU acceleration for Ollama, refer to the GPU setup instructions in the `docker-compose.yml` file.*

---

## 💡 How it Works (Under the Hood)

### 1. Document Parsing & Image Extraction
When a PDF is uploaded, **Docling** parses the document structure. It extracts text sections, markdown tables, and crops out figures/drawings. The coordinates of these elements are stored, and pictures are saved to the `extracted_images/` folder.

### 2. Hierarchical Chunking
Instead of splitting text using arbitrary character limits, the system uses Docling's **Hierarchical Chunker** to group text semantically under its corresponding parent headings. Large chunks are then split with overlapping sliding windows.

### 3. Vector Embedding & Retrieval
Each chunk is embedded using `nomic-embed-text` with a prefix task (`search_document: `) and stored in **ChromaDB**. When a query is made, it is embedded (`search_query: `) and a cosine similarity search retrieves the top relevant context blocks.

### 4. Grounded Synthesis
The retrieved sections, table markdown, and any extracted page figures are sent to `qwen2.5:7b-instruct` inside a safety-oriented system prompt. The model synthesizes the answer, citing the exact page and section, and displays the related diagrams in the chat.
