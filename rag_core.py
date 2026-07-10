import os
import hashlib
import chromadb
import ollama
import json
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load configuration from .env file
load_dotenv()

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc, 
            table_serializer=MarkdownTableSerializer(),
        )

# Load configuration from environment
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "chroma_db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def get_ollama_client():
    """Initialize and return the Ollama API client."""
    return ollama.Client(host=OLLAMA_HOST)

def get_chroma_client():
    """Initialize and return the local persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def check_ollama_connection():
    """Verify if the Ollama service is running and reachable."""
    client = get_ollama_client()
    try:
        client.list()
        return True
    except Exception:
        return False

def ensure_models_pulled(models=["qwen2.5:7b-instruct", "nomic-embed-text"]):
    """
    Check if the required models are pulled in Ollama.
    Yields (model_name, progress_dict) if a pull is active.
    """
    client = get_ollama_client()
    existing_models = []
    try:
        model_list = client.list()
        for m in model_list.get("models", []):
            existing_models.append(m.get("name", ""))
            existing_models.append(m.get("model", ""))
    except Exception as e:
        print(f"Error listing models: {e}")

    for model in models:
        exists = False
        for ex in existing_models:
            # Check for exact match or tags like :latest
            if ex == model or ex.startswith(model + ":") or model.startswith(ex + ":"):
                exists = True
                break
        if not exists:
            # Yield pull status generator
            for progress in client.pull(model=model, stream=True):
                yield model, progress

def split_text_with_overlap(text, max_chars=1500, overlap=250):
    """
    Split text into overlapping chunks of max_chars length, trying to split
    on natural word/sentence boundaries.
    """
    if len(text) <= max_chars:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to find a good split point in the last 150 characters
        split_pos = -1
        for sep in ["\n\n", "\n", " ", ". "]:
            pos = text.rfind(sep, start + max_chars - 150, end)
            if pos != -1:
                split_pos = pos + len(sep)
                break
                
        if split_pos == -1:
            split_pos = end
            
        chunks.append(text[start:split_pos])
        start = split_pos - overlap
        if start < 0:
            start = 0
            
    return chunks


def parse_and_chunk_pdf(file_path, page_range=None):
    """
    Parse a PDF using Docling, export markdown, extract images,
    and split into chunks using layout-aware HierarchicalChunker.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

    # 1. Setup options to generate images and enable CUDA acceleration
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True  # Enable figure extraction
    pipeline_options.images_scale = 2.0  # High-quality resolution
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"Docling converting document: {file_path}")
    if page_range:
        result = converter.convert(file_path, page_range=page_range)
    else:
        result = converter.convert(file_path)
    print("Docling conversion complete.")

    # Save full document markdown file alongside the original PDF
    pdf_path = Path(file_path)
    md_path = pdf_path.with_suffix(".md")
    try:
        markdown_text = result.document.export_to_markdown()
        md_path.write_text(markdown_text, encoding="utf-8")
        print(f"Saved parsed markdown representation to: {md_path}")
    except Exception as e:
        print(f"Error saving markdown file: {e}")

    # 2. Extract and save each figure
    pdf_base = pdf_path.stem
    output_dir = Path("extracted_images") / pdf_base
    output_dir.mkdir(parents=True, exist_ok=True)

    image_records = []
    image_counter = 0
    image_mapping = {}

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            image_counter += 1
            img_filename = f"figure_{image_counter}.png"
            img_path = output_dir / img_filename
            
            try:
                # Save image to disk
                element.get_image(result.document).save(img_path, "PNG")
                
                # Find which page it was on
                page_num = element.prov[0].page_no if element.prov else "unknown"
                page_str = str(page_num)
                
                rel_path = f"extracted_images/{pdf_base}/{img_filename}"
                if page_str not in image_mapping:
                    image_mapping[page_str] = []
                image_mapping[page_str].append(rel_path)
                
                image_records.append({
                    "image_path": str(img_path),
                    "page": page_num
                })
            except Exception as img_err:
                print(f"Error extracting image {image_counter}: {img_err}")

    # Save the mapping metadata to metadata.json
    try:
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(image_mapping, f, indent=2)
        print(f"Saved image mapping to {output_dir / 'metadata.json'}")
    except Exception as e:
        print(f"Error saving metadata.json: {e}")

    print(f"Extracted {image_counter} images.")

    # 3. Chunk the document using HierarchicalChunker
    print("Chunking document...")
    chunker = HierarchicalChunker(serializer_provider=MDTableSerializerProvider())
    doc_chunks = list(chunker.chunk(result.document))
    print(f"Generated {len(doc_chunks)} raw layout-aware chunks.")

    # 4. Merge adjacent layout-aware chunks under the same heading section
    print("Merging adjacent layout-aware chunks...")
    merged_docs = []
    current_chunk = None

    for dc in doc_chunks:
        # Determine page range from doc_items
        page_nos = set()
        for item in dc.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for p in item.prov:
                    if hasattr(p, "page_no") and p.page_no:
                        page_nos.add(p.page_no)
        
        if page_nos:
            page_start = min(page_nos)
            page_end = max(page_nos)
        else:
            page_start = 1
            page_end = 1

        section = " > ".join(dc.meta.headings) if dc.meta.headings else "General"
        text = dc.text.strip()
        if not text:
            continue
        if current_chunk is None:
            current_chunk = {
                "text": text,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "page_nos": page_nos if page_nos else {1}
            }
        else:
            same_section = (current_chunk["section"] == section)
            # Combine up to 1500 characters
            can_fit = (len(current_chunk["text"]) + len(text) + 2 <= 1500)

            if same_section and can_fit:
                current_chunk["text"] += "\n\n" + text
                current_chunk["page_nos"].update(page_nos)
                current_chunk["page_start"] = min(current_chunk["page_nos"])
                current_chunk["page_end"] = max(current_chunk["page_nos"])
            else:
                merged_docs.append(current_chunk)
                current_chunk = {
                    "text": text,
                    "section": section,
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_nos": page_nos if page_nos else {1}
                }

    if current_chunk:
        merged_docs.append(current_chunk)

    print(f"Merged raw chunks into {len(merged_docs)} structural sections.")

    # 5. Split large chunks using sliding window character splitter
    chunks = []
    filename = pdf_path.name

    for item in merged_docs:
        sub_texts = split_text_with_overlap(item["text"], max_chars=1500, overlap=200)
        for sub_text in sub_texts:
            page_start = item["page_start"]
            page_end = item["page_end"]
            pages_str = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
            
            # Prepend the section name to the text to keep parent context intact in all sub-chunks
            full_text = f"Section: {item['section']}\n\n{sub_text}"
            
            chunks.append({
                "text": full_text,
                "metadata": {
                    "source": filename,
                    "section": item["section"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "pages": pages_str
                }
            })

    print(f"Generated {len(chunks)} final overlapping context chunks.")
    return chunks

def ingest_pdf_to_chroma(file_path, collection_name="siemens_manuals", page_range=None, progress_callback=None):
    """
    Parse, chunk, calculate local embeddings, and save PDF into ChromaDB.
    Updates progress if progress_callback is provided.
    """
    if progress_callback:
        progress_callback(0.0, "Parsing document with Docling (extracting layout & figures)...")

    chunks = parse_and_chunk_pdf(file_path, page_range=page_range)
    if not chunks:
        return 0

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    ollama_client = get_ollama_client()

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        content = chunk["text"]
        meta = chunk["metadata"]
        
        # Calculate unique ID based on content hash to prevent duplicates
        chunk_id = hashlib.md5(f"{meta['source']}_{i}_{content}".encode("utf-8")).hexdigest()
        
        if progress_callback:
            # We scale embeddings generation to 0.1 - 0.9 range
            progress_val = 0.1 + (float(i) / total) * 0.8
            progress_callback(progress_val, f"Generating embedding for chunk {i+1} of {total}...")

        # Calculate local embedding via Ollama with task prefix
        response = ollama_client.embeddings(model="nomic-embed-text", prompt="search_document: " + content)
        embedding = response["embedding"]

        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(meta)
        embeddings.append(embedding)

    if progress_callback:
        progress_callback(0.95, f"Saving {total} embeddings to local ChromaDB...")

    # Insert into ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    if progress_callback:
        progress_callback(1.0, f"Successfully indexed all {total} chunks!")

    # Invalidate document list cache
    try:
        list_ingested_documents.clear()
    except Exception:
        pass

    return len(chunks)

def generate_llm_response(model_name, system_prompt, user_prompt, chat_history=None):
    """
    Generate response from the selected LLM.
    If chat_history is provided, it passes the full conversation context to the LLM (for generation).
    """
    messages = []
    
    # If chat_history is provided, convert and append it (excluding sources to keep payload clean)
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    # Append the current prompt
    if model_name == "ChatGPT (GPT-4o)":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API Key is missing. Please configure OPENAI_API_KEY in your .env file."
            )
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)
        api_messages.append({"role": "user", "content": user_prompt})
        
        chat_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=api_messages
        )
        return chat_response.choices[0].message.content

    elif model_name == "Claude (Claude 3.5 Sonnet)":
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "Anthropic API Key is missing. Please configure ANTHROPIC_API_KEY in your .env file."
            )
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        api_messages = list(messages)
        api_messages.append({"role": "user", "content": user_prompt})
        
        chat_response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system_prompt if system_prompt else "",
            messages=api_messages
        )
        return chat_response.content[0].text

    else:
        # Default / Fallback to local Ollama (Qwen 2.5)
        client = get_ollama_client()
        
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)
        api_messages.append({"role": "user", "content": user_prompt})
        
        chat_response = client.chat(
            model="qwen2.5:7b-instruct",
            messages=api_messages
        )
        return chat_response["message"]["content"]


def reformulate_query(query, model_name, chat_history):
    """
    Given the latest user question and the conversation history, reformulate it into
    a standalone search query.
    """
    if not chat_history or len(chat_history) == 0:
        return query
        
    history_str = ""
    # Use last 5 turns to keep context brief but sufficient
    for msg in chat_history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    system_prompt = (
        "You are a helpful assistant that reformulates follow-up questions into standalone search queries."
    )
    user_prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question (in English) that "
        "contains all necessary context (e.g. specific product names like CPU 410-5H) "
        "so it can be used for search in a vector database.\n"
        "Do NOT answer the question. Just return the rephrased standalone question and nothing else. "
        "If the question is already fully standalone and does not reference any pronouns or implicit context from the history, "
        "return it exactly as-is.\n\n"
        f"Conversation History:\n{history_str}\n"
        f"Follow-up Question: {query}\n"
        "Standalone Question:"
    )
    try:
        standalone = generate_llm_response(model_name, system_prompt, user_prompt)
        standalone = standalone.strip().strip('"').strip("'")
        return standalone
    except Exception as e:
        print(f"Error reformulating query: {e}")
        return query


def query_rag(query, model_name="Ollama (Qwen 2.5)", collection_name="siemens_manuals", n_results=5, chat_history=None):
    """
    Perform semantic search over ChromaDB and synthesize response using the selected model with chat history context.
    """
    # 1. Clean history to get only past turns (exclude the current user turn if already added)
    history_turns = []
    if chat_history:
        if chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == query:
            history_turns = chat_history[:-1]
        else:
            history_turns = chat_history

    # 2. Reformulate query to be standalone (for high precision ChromaDB retrieval)
    standalone_query = reformulate_query(query, model_name, history_turns)

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    ollama_client = get_ollama_client()

    # Get local embedding for standalone query with task prefix
    response = ollama_client.embeddings(model="nomic-embed-text", prompt="search_query: " + standalone_query)
    query_embedding = response["embedding"]

    # Perform ChromaDB search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Reconstruct contexts and source records
    retrieved_chunks = []
    contexts = []
    
    if results and results.get("documents") and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, doc_text in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            dist = distances[idx] if idx < len(distances) else 0.0
            
            contexts.append(
                f"[Source: {meta.get('source', 'Unknown')}, Page(s): {meta.get('pages', 'N/A')}, Section: {meta.get('section', 'N/A')}]\n{doc_text}"
            )
            
            retrieved_chunks.append({
                "text": doc_text,
                "source": meta.get("source", "Unknown"),
                "pages": meta.get("pages", "N/A"),
                "section": meta.get("section", "N/A"),
                "score": 1.0 - dist  # Approximate similarity score
            })

    context_str = "\n\n---\n\n".join(contexts)

    # Compile domain-specific system instruction
    system_prompt = (
        "You are an expert Siemens Industrial Automation engineer specializing in SIMATIC controllers (S7-300, S7-400, S7-1200, S7-1500) and PLC hardware configurations.\n"
        "Your task is to answer the user's technical question based strictly on the provided Siemens product manual context.\n"
        "Guidelines:\n"
        "1. GROUNDING: Answer the question using ONLY the provided manual context. Do not invent details. If the context does not contain the answer, state clearly: 'The provided Siemens manual context does not contain information to answer this question.'\n"
        "2. SAFETY: If the context contains 'DANGER', 'WARNING', 'CAUTION', or 'NOTICE' statements regarding wiring, electrical loads, or automation operation, you MUST emphasize them in bold in your response.\n"
        "3. PRECISION: Provide specific section and page number references in your answer when citing the manual.\n"
        "4. CODE: Format any Structured Control Language (SCL), Statement List (STL), or configuration examples in markdown block format."
    )

    user_prompt = f"Siemens Manual Context:\n{context_str}\n\nUser Question:\n{query}"

    # Generate answer using selected LLM (passing past conversation history)
    answer = generate_llm_response(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        chat_history=history_turns
    )

    return answer, retrieved_chunks

def sanitize_collection_name(name):
    """Sanitize collection name to conform to ChromaDB naming rules."""
    import re
    # Convert to lowercase and trim spaces
    name = name.lower().strip()
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove any characters that are not alphanumeric, underscore, dot, or hyphen
    name = re.sub(r'[^a-z0-9._-]', '', name)
    # Remove consecutive dots
    name = re.sub(r'\.+', '.', name)
    # Ensure it starts with alphanumeric
    name = re.sub(r'^[^a-z0-9]+', '', name)
    # Ensure it ends with alphanumeric
    name = re.sub(r'[^a-z0-9]+$', '', name)
    # Ensure it is at least 3 characters
    if len(name) < 3:
        name = (name + "col")[:3]
    # Ensure it is at most 63 characters
    return name[:63]

def list_collections():
    """List all collection names in ChromaDB."""
    chroma_client = get_chroma_client()
    collections = chroma_client.list_collections()
    return sorted([c.name for c in collections])


@st.cache_data
def list_ingested_documents(collection_name="siemens_manuals"):
    """List all unique document names currently in the vector store."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    data = collection.get(include=["metadatas"])
    if not data or not data.get("metadatas"):
        return []
    
    sources = set()
    for meta in data["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sorted(list(sources))

def delete_document(document_name, collection_name="siemens_manuals"):
    """Remove all chunks associated with a specific manual from the vector database."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.delete(where={"source": document_name})
    
    # Invalidate document list cache
    try:
        list_ingested_documents.clear()
    except Exception:
        pass
        
    return True


def get_images_for_pages(source_name, pages_str):
    """
    Resolve and return paths to any extracted images matching the pages.
    """
    pdf_base = os.path.splitext(source_name)[0]
    meta_path = os.path.join("extracted_images", pdf_base, "metadata.json")
    if not os.path.exists(meta_path):
        return []
        
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        return []
        
    pages = []
    if "-" in str(pages_str):
        try:
            start, end = map(int, str(pages_str).split("-"))
            pages = list(range(start, end + 1))
        except ValueError:
            pass
    else:
        try:
            pages = [int(pages_str)]
        except ValueError:
            pass
            
    images = []
    for p in pages:
        p_str = str(p)
        if p_str in mapping:
            for img_rel in mapping[p_str]:
                # verify it exists
                if os.path.exists(img_rel):
                    images.append(img_rel)
    return images
