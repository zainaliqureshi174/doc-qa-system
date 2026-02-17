# core/document_processor.py
# Handles loading documents of different formats and splitting
# them into chunks for vector storage and retrieval.

import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from the given file path.
    Automatically detects file type and uses the appropriate loader.

    Supported formats: PDF, DOCX, TXT

    Args:
        file_path: Full path to the document file

    Returns:
        List of Document objects with content and metadata
    """
    # Get the file extension in lowercase
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    # Select the appropriate loader based on file type
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(
            f"Unsupported file format: '{extension}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )

    # Load and return the documents
    documents = loader.load()
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks for
    better retrieval accuracy.

    Uses RecursiveCharacterTextSplitter which intelligently
    splits on paragraphs, then sentences, then words —
    preserving context as much as possible.

    Args:
        documents: List of Document objects to split

    Returns:
        List of smaller Document chunks with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Maximum characters per chunk
        chunk_overlap=200,     # Overlap between consecutive chunks
        length_function=len,   # Use character count for length
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def process_document(file_path: str) -> List[Document]:
    """
    Complete pipeline: loads a document and splits it into chunks.
    This is the main function called by the rest of the application.

    Args:
        file_path: Full path to the document file

    Returns:
        List of Document chunks ready for embedding
    """
    # Step 1: Load the document
    documents = load_document(file_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    return chunks
