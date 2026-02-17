# app.py
# Main Streamlit application for the Document Q&A System.
# Run with: streamlit run app.py

import streamlit as st
from core.document_processor import process_document
from core.vector_store import create_vector_store, add_documents_to_store
from core.qa_chain import build_qa_chain, get_answer, create_memory
from core.summarizer import summarize_documents
from utils.helpers import get_available_models
import tempfile
import os


# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
def initialize_session_state():
    """Initialize all session state variables if not already set."""

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "current_chunks" not in st.session_state:
        st.session_state.current_chunks = []


initialize_session_state()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def process_uploaded_file(uploaded_file) -> bool:
    """
    Processes an uploaded file and adds it to the vector store.
    Returns True if successful, False otherwise.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process document into chunks
        with st.spinner(f"Processing {uploaded_file.name}..."):
            chunks = process_document(tmp_path)

        # Add to vector store
        with st.spinner("Creating embeddings and updating vector store..."):
            st.session_state.vector_store = add_documents_to_store(
                chunks=chunks,
                existing_store=st.session_state.vector_store
            )
            st.session_state.current_chunks.extend(chunks)

        # Rebuild QA chain with updated vector store
        st.session_state.memory = create_memory()
        st.session_state.qa_chain = build_qa_chain(
            vector_store=st.session_state.vector_store,
            memory=st.session_state.memory
        )

        # Track uploaded file name
        st.session_state.uploaded_files.append(uploaded_file.name)

        # Clean up temp file
        os.unlink(tmp_path)

        return True

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False


def clear_chat():
    """Clears the chat history and resets memory."""
    st.session_state.chat_history = []
    st.session_state.memory = create_memory()

    # Rebuild chain with fresh memory
    if st.session_state.vector_store:
        st.session_state.qa_chain = build_qa_chain(
            vector_store=st.session_state.vector_store,
            memory=st.session_state.memory
        )


def reset_all():
    """Resets everything — clears all documents and chat."""
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.memory = None
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []
    st.session_state.current_chunks = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Q&A")
    st.markdown("---")

    # Model selector
    st.subheader("🤖 Model Selection")
    available_models = get_available_models()
    selected_model_label = st.selectbox(
        "Choose LLM Model",
        options=list(available_models.keys()),
        index=0
    )
    selected_model = available_models[selected_model_label]

    st.markdown("---")

    # File uploader
    st.subheader("📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="You can upload multiple documents one by one"
    )

    if uploaded_file:
        if uploaded_file.name not in st.session_state.uploaded_files:
            success = process_uploaded_file(uploaded_file)
            if success:
                st.success(f"✅ {uploaded_file.name} processed")
        else:
            st.info(f"'{uploaded_file.name}' is already loaded")

    # Show loaded documents
    if st.session_state.uploaded_files:
        st.markdown("**Loaded Documents:**")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- 📄 {fname}")

    st.markdown("---")

    # Summarize button
    if st.session_state.current_chunks:
        st.subheader("📝 Document Summary")
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = summarize_documents(
                    st.session_state.current_chunks,
                    model_name=selected_model
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**Document Summary:**\n\n{summary}",
                    "sources": []
                })

    st.markdown("---")

    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            reset_all()
            st.rerun()


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("📄 Document Q&A System")
st.markdown(
    "Upload documents and ask questions. "
    "The system will find answers from your documents."
)
st.markdown("---")

# Show welcome message if no documents loaded
if not st.session_state.uploaded_files:
    st.info(
        "👈 Upload a document from the sidebar to get started. "
        "Supported formats: PDF, DOCX, TXT"
    )

# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])

    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

            # Show source citations if available
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(
                            f"> {source.page_content[:300]}..."
                            if len(source.page_content) > 300
                            else f"> {source.page_content}"
                        )
                        if source.metadata.get("source"):
                            fname = os.path.basename(
                                source.metadata["source"]
                            )
                            st.caption(f"📄 File: {fname}")
                        if source.metadata.get("page") is not None:
                            st.caption(
                                f"📖 Page: {source.metadata['page'] + 1}"
                            )
                        st.markdown("---")


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if st.session_state.qa_chain:
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get answer from QA chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_answer(
                    st.session_state.qa_chain,
                    user_question
                )
                answer = result["answer"]
                sources = result["source_documents"]

            st.markdown(answer)

            # Show sources
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(
                            f"> {source.page_content[:300]}..."
                            if len(source.page_content) > 300
                            else f"> {source.page_content}"
                        )
                        if source.metadata.get("source"):
                            fname = os.path.basename(
                                source.metadata["source"]
                            )
                            st.caption(f"📄 File: {fname}")
                        if source.metadata.get("page") is not None:
                            st.caption(
                                f"📖 Page: {source.metadata['page'] + 1}"
                            )
                        st.markdown("---")

        # Save assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    # Show disabled input when no documents loaded
    st.chat_input(
        "Upload a document first to start asking questions...",
        disabled=True
    )
