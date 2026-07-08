import streamlit as st
import rag_core

def render_sidebar():
    with st.sidebar:
        st.image("materials/cnautomation-logo.jpg", use_container_width=True)

        if st.button("Reference Chat", use_container_width=True):
            st.switch_page("app.py")

        if st.button("Ingest & Manage Documents", use_container_width=True):
            st.switch_page("pages/ingest_page.py")
    # Title & Styling
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stSidebarCollapseButton"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Connection Check
    ollama_status = rag_core.check_ollama_connection()

    if ollama_status:
        st.sidebar.markdown("### Model Pre-checks")
        st.sidebar.success("Ollama Connection: Online")

        # Check/Pull Models in Sidebar if needed

        # Pull status container in sidebar
        status_container = st.sidebar.empty()

        # Check if models are present
        try:
            models_to_check = ["qwen2.5:7b-instruct", "nomic-embed-text"]
            pull_generators = list(rag_core.ensure_models_pulled(models_to_check))
            if len(pull_generators) > 0:
                status_container.warning("Downloading missing models...")
                progress_bar = st.sidebar.progress(0.0)

                for model_name, progress in rag_core.ensure_models_pulled(models_to_check):
                    # Format progress
                    status_text = progress.get("status", "Downloading...")
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 1)

                    # Update progress bar
                    pct = float(completed) / float(total) if total > 0 else 0.0
                    progress_bar.progress(min(pct, 1.0))
                    status_container.info(f"{model_name}: {status_text}")

                progress_bar.empty()
                status_container.success("All models downloaded!")
            else:
                status_container.success("Models loaded locally")
        except Exception as e:
            status_container.error(f"Error checking models: {str(e)}")
    else:
        st.sidebar.error("Ollama Connection: Offline")
        st.sidebar.warning(
            f"Could not connect to Ollama at `{rag_core.OLLAMA_HOST}`.\n\n"
            "Please start Ollama locally (`ollama serve`) or run the docker environment."
        )




