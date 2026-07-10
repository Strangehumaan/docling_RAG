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

    import time

    # Initialize connection cache in session state
    if "ollama_status_cache" not in st.session_state:
        st.session_state["ollama_status_cache"] = {"status": False, "last_checked": 0.0}
        
    cache = st.session_state["ollama_status_cache"]
    current_time = time.time()
    
    # Check connection at most once every 10 seconds
    if current_time - cache["last_checked"] > 10.0:
        try:
            ollama_status = rag_core.check_ollama_connection()
        except Exception:
            ollama_status = False
        st.session_state["ollama_status_cache"] = {"status": ollama_status, "last_checked": current_time}
    else:
        ollama_status = cache["status"]

    if ollama_status:
        st.sidebar.markdown("### Model Pre-checks")
        st.sidebar.success("Ollama Connection: Online")

        # Pull status container in sidebar
        status_container = st.sidebar.empty()

        # Initialize checked models set in session state
        if "checked_models" not in st.session_state:
            st.session_state["checked_models"] = set()

        # Determine which models need to be verified based on selected LLM
        models_to_check = ["nomic-embed-text"]
        if st.session_state.get("selected_model", "Ollama (Qwen 2.5)") == "Ollama (Qwen 2.5)":
            models_to_check.append("qwen2.5:7b-instruct")

        # Find models that haven't been successfully verified yet in this session
        unchecked_models = [m for m in models_to_check if m not in st.session_state["checked_models"]]

        if unchecked_models:
            # Check/Pull Models in Sidebar
            try:
                pull_generators = list(rag_core.ensure_models_pulled(unchecked_models))
                if len(pull_generators) > 0:
                    status_container.warning("Downloading missing models...")
                    progress_bar = st.sidebar.progress(0.0)

                    for model_name, progress in rag_core.ensure_models_pulled(unchecked_models):
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
                
                # Mark as verified
                st.session_state["checked_models"].update(unchecked_models)
                status_container.success("Models loaded locally")
            except Exception as e:
                status_container.error(f"Error checking models: {str(e)}")
        else:
            status_container.success("Models loaded locally")
    else:
        st.sidebar.error("Ollama Connection: Offline")
        st.sidebar.warning(
            f"Could not connect to Ollama at `{rag_core.OLLAMA_HOST}`.\n\n"
            "Please start Ollama locally (`ollama serve`) or run the docker environment."
        )




