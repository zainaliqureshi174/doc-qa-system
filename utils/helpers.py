# utils/helpers.py
# This module handles configuration loading and shared utility functions

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_groq_api_key() -> str:
    """
    Retrieves the Groq API key from environment variables.
    Raises a clear error if the key is missing so the user
    knows exactly what to fix.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add your Groq API key to the .env file."
        )

    return api_key


def get_available_models() -> dict:
    """
    Returns a dictionary of available LLM models.
    Keeping models in one place makes it easy to add
    or remove options without touching other files.
    """
    return {
        "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B (Faster)": "llama-3.1-8b-instant",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
    }
