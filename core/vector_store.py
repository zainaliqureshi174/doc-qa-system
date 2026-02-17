# core/vector_store.py
# Handles creating vector embeddings from document chunks
# and storing/retrieving them using FAISS.

import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Path where the FAISS index will be saved on disk
FAISS_INDEX_PATH = "faiss_index"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads and returns the HuggingFace embedding model.
    Uses all-MiniLM-L6-v2 — a lightweight but powerful
    model that runs locally with no API key required.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Creates a FAISS vector store from document chunks.
    Converts each chunk to an embedding and indexes it.

    Args:
        chunks: List of Document chunks from document_processor

    Returns:
        FAISS vector store ready for similarity search
    """
    # Load the embedding model
    embeddings = get_embedding_model()

    # Create FAISS index from document chunks
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save the index to disk so it persists between sessions
    vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """
    Loads an existing FAISS index from disk if it exists.
    Returns None if no index has been created yet.

    Returns:
        FAISS vector store or None
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return None

    embeddings = get_embedding_model()

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def add_documents_to_store(
    chunks: List[Document],
    existing_store: Optional[FAISS] = None
) -> FAISS:
    """
    Adds new document chunks to an existing vector store
    or creates a new one if none exists.
    This powers the multi-document support feature.

    Args:
        chunks: New document chunks to add
        existing_store: Existing FAISS store to add to (optional)

    Returns:
        Updated FAISS vector store
    """
    embeddings = get_embedding_model()

    if existing_store is None:
        # No existing store — create a fresh one
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    else:
        # Add new chunks to the existing store
        vector_store = existing_store
        vector_store.add_documents(chunks)

    # Save updated index to disk
    vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store


def get_retriever(vector_store: FAISS, k: int = 4):
    """
    Creates a retriever from the vector store.
    The retriever fetches the top-k most relevant chunks
    for any given query.

    Args:
        vector_store: FAISS vector store
        k: Number of chunks to retrieve (default 4)

    Returns:
        LangChain retriever object
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever
