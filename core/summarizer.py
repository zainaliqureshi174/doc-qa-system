# core/summarizer.py
# Handles document summarization using LangChain's
# map_reduce chain — works reliably on large documents.

from typing import List
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from core.qa_chain import get_llm


def summarize_documents(
    chunks: List[Document],
    model_name: str = None
) -> str:
    """
    Generates a comprehensive summary of the provided
    document chunks using map_reduce strategy.

    Map step: each chunk is summarized individually.
    Reduce step: all summaries are combined into one.

    Args:
        chunks: List of Document chunks to summarize
        model_name: Optional Groq model name to use

    Returns:
        A clean, comprehensive summary string
    """
    # Initialize the LLM
    llm = get_llm(model_name)

    # Prompt for summarizing each individual chunk (map step)
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Write a concise summary of the following text:

{text}

Concise Summary:"""
    )

    # Prompt for combining all chunk summaries (reduce step)
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""You are given a series of summaries from different 
parts of a document. Combine them into one comprehensive, 
well-structured final summary.

Summaries:
{text}

Final Comprehensive Summary:"""
    )

    # Load the map_reduce summarization chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False
    )

    # Run the chain and return the summary
    result = summary_chain.invoke({"input_documents": chunks})

    return result["output_text"]
