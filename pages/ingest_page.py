import streamlit as st
import os
from UI import sidebar
import rag_core
st.set_page_config(
    page_title="CN RAG Assistant",
    page_icon="materials/title_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Hide the default multipage navigation */
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

sidebar.render_sidebar()

# with st.sidebar:
#     st.image("materials/cnautomation-logo.jpg", use_container_width=True)
#
#     if st.button("Reference Chat", use_container_width=True):
#         st.switch_page("pages/app.py")
#
#     if st.button("Ingest & Manage Documents", use_container_width=True):
#         st.switch_page("pages/ingest_page.py")
st.title("Document Ingestion & Management")
st.markdown(
    "Upload Siemens manuals to extract text and tables using **IBM Docling**, chunk them, and store their vector representations in the local database."
)

# Custom local path from user
DEFAULT_MANUAL_PATH = r"C:\Users\saadn\Documents\CN Automation\RAG_System\manual\AS410-5H_System_Manuals7400_cpu_410_proc_autom_smart_system_system_en-US_en-US.pdf"

# Layout: Two columns (Upload/Ingest vs. Current Database)
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("Ingest New Manual")
    
    # Collection selection / creation UI
    try:
        collections = rag_core.list_collections()
    except Exception as e:
        collections = []
        st.error(f"Error listing collections: {e}")
        
    if not collections:
        collections = ["siemens_manuals"]
        
    collection_options = collections + ["[Create New Collection...]"]
    
    # Default index to 's7-1200' or 'siemens_manuals'
    default_idx = 0
    if "s7-1200" in collection_options:
        default_idx = collection_options.index("s7-1200")
    elif "siemens_manuals" in collection_options:
        default_idx = collection_options.index("siemens_manuals")
        
    selected_col_option = st.selectbox(
        "Target Collection:",
        options=collection_options,
        index=default_idx,
        help="Select which collection to ingest this manual into. Use model-specific collections (e.g. s7-1200) for better RAG grounding."
    )
    
    target_collection = ""
    if selected_col_option == "[Create New Collection...]":
        new_col_name = st.text_input("New Collection Name:", value="", placeholder="e.g. s7-1200")
        if new_col_name:
            target_collection = rag_core.sanitize_collection_name(new_col_name)
            st.caption(f"Sanitized Collection Name: `{target_collection}`")
        else:
            st.warning("Please enter a collection name.")
    else:
        target_collection = selected_col_option

    # Enable ingestion only if target_collection is specified
    is_ready = bool(target_collection)

    # 1. Local Auto-Detection (Premium UX)
    if os.path.exists(DEFAULT_MANUAL_PATH):
        st.info("**Local Siemens Manual Detected**")
        st.markdown(
            f"Found CPU 410 System Manual at:\n`{DEFAULT_MANUAL_PATH}`"
        )
        if st.button("Ingest Local File Direct", type="primary", key="ingest_local", disabled=not is_ready):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.info(msg)

            try:
                num_chunks = rag_core.ingest_pdf_to_chroma(
                    DEFAULT_MANUAL_PATH, 
                    collection_name=target_collection,
                    progress_callback=update_progress
                )
                status_text.empty()
                progress_bar.empty()
                st.success(f"Successfully ingested local manual into `{target_collection}`! Split into {num_chunks} chunks.")
                st.rerun()
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Failed to ingest local file: {str(e)}")
        st.markdown("---")

    # 2. Regular File Uploader
    uploaded_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
    if uploaded_file is not None:
        st.write(f"Filename: `{uploaded_file.name}`")
        if st.button("Ingest Uploaded PDF", type="secondary", disabled=not is_ready):
            # Ensure upload folder exists
            temp_dir = "./temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.info(msg)
                
            try:
                num_chunks = rag_core.ingest_pdf_to_chroma(
                    temp_path, 
                    collection_name=target_collection,
                    progress_callback=update_progress
                )
                status_text.empty()
                progress_bar.empty()
                st.success(f"Successfully processed `{uploaded_file.name}` into `{target_collection}`! Created {num_chunks} chunks.")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.rerun()
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Error parsing PDF: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with col2:
    st.subheader("Database Library")
    
    # Dropdown to filter library view by collection
    try:
        lib_collections = rag_core.list_collections()
    except Exception:
        lib_collections = ["siemens_manuals"]
        
    if not lib_collections:
        lib_collections = ["siemens_manuals"]
        
    selected_lib_col = st.selectbox(
        "View Collection:",
        options=lib_collections,
        help="Select which collection to view or manage."
    )
    
    # List ingested documents in the selected collection
    try:
        docs = rag_core.list_ingested_documents(collection_name=selected_lib_col)
        if not docs:
            st.info(f"The collection `{selected_lib_col}` is currently empty. Please ingest a manual to begin.")
        else:
            st.write(f"Ingested Manuals ({len(docs)}):")
            for doc in docs:
                with st.container(border=True):
                    c_doc, c_btn = st.columns([4, 1])
                    c_doc.markdown(f"**{doc}**")
                    if c_btn.button("🗑️", key=f"del_{selected_lib_col}_{doc}", help=f"Delete {doc} from {selected_lib_col}"):
                        with st.spinner("Removing document..."):
                            rag_core.delete_document(doc, collection_name=selected_lib_col)
                            st.success(f"Deleted `{doc}` from `{selected_lib_col}`")
                            st.rerun()
            
            st.markdown("---")
            
            # Action: Delete selected collection
            if st.button(f"Delete Collection '{selected_lib_col}'", type="secondary", help="Delete this collection and all its documents"):
                with st.spinner("Deleting collection..."):
                    client = rag_core.get_chroma_client()
                    try:
                        client.delete_collection(selected_lib_col)
                        # Clear document list cache
                        rag_core.list_ingested_documents.clear()
                        st.success(f"Collection `{selected_lib_col}` successfully deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting collection: {e}")
            
            # Action: Wipe entire database
            if st.button("Wipe Entire Database", type="secondary", help="Wipe all collections in the database"):
                with st.spinner("Wiping collections..."):
                    client = rag_core.get_chroma_client()
                    try:
                        all_cols = client.list_collections()
                        for col in all_cols:
                            client.delete_collection(col.name)
                        # Clear document list cache
                        rag_core.list_ingested_documents.clear()
                        st.success("Vector database successfully wiped.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error wiping database: {e}")
    except Exception as e:
        st.error(f"Error loading database index: {str(e)}")
