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
st.title("Technical Reference Chat")
st.markdown(
    "Query your local manuals library. Responses are grounded in retrieved context using local **Qwen2.5:7b-instruct**."
)

# 1. Database Check
try:
    ingested_docs = rag_core.list_ingested_documents()
except Exception as e:
    st.error(f"Error accessing database: {str(e)}")
    ingested_docs = []

if not ingested_docs:
    st.warning("No documents have been ingested yet. Please go to **Ingest & Manage Documents** in the sidebar to load your Siemens manuals.")
else:
    # Optional: Select specific manual or search all
    st.sidebar.subheader("Chat Configuration")
    target_docs = st.sidebar.multiselect(
        "Reference scope:",
        options=ingested_docs,
        default=ingested_docs,
        help="Limit queries to selected manuals only. Defaults to all."
    )
    
    num_sources = st.sidebar.slider("Number of retrieved chunks:", min_value=1, max_value=10, value=5)
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Clear conversation button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(" Grounding Sources"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(
                            f"**Source {i+1}**: {source['source']} | "
                            f"**Section**: {source['section']} | "
                            f"**Page(s)**: {source['pages']}"
                        )
                        st.code(source["text"], language="markdown")
                        
                        # Show any extracted images for this page
                        image_paths = rag_core.get_images_for_pages(source["source"], source["pages"])
                        if image_paths:
                            st.markdown("**Related Figures:**")
                            for img_path in image_paths:
                                st.image(img_path, caption=os.path.basename(img_path))
                        
    # Input field
    if prompt := st.chat_input("Ask a technical or safety question... (e.g. 'What is the diagnostic byte format for CPU 410?')"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Query local RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching local vector index & generating response..."):
                try:
                    # Execute query
                    # Note: We can filter sources by selected target_docs if needed,
                    # but simple query_rag will query the collection.
                    answer, sources = rag_core.query_rag(prompt, n_results=num_sources)
                    
                    # Filter sources if user selected a subset of docs
                    if target_docs:
                        sources = [s for s in sources if s["source"] in target_docs]
                        
                    st.markdown(answer)
                    
                    # Display references
                    if sources:
                        with st.expander("Grounding Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(
                                    f"**Source {i+1}**: {source['source']} | "
                                    f"**Section**: {source['section']} | "
                                    f"**Page(s)**: {source['pages']}"
                                )
                                st.code(source["text"], language="markdown")
                                
                                # Show any extracted images for this page
                                image_paths = rag_core.get_images_for_pages(source["source"], source["pages"])
                                if image_paths:
                                    st.markdown("**Related Figures:**")
                                    for img_path in image_paths:
                                        st.image(img_path, caption=os.path.basename(img_path))
                                
                    # Save response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error executing RAG pipeline: {str(e)}")
                    st.info("Ensure Ollama is running and the models are fully downloaded.")
