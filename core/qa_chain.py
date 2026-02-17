# core/qa_chain.py
# Builds the conversational QA chain that connects the
# retriever, memory, and LLM together.

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from utils.helpers import get_groq_api_key, get_available_models


def get_llm(model_name: str = None) -> ChatGroq:
    """
    Initializes and returns the Groq LLM.
    Defaults to Llama 3.3 70B if no model is specified.

    Args:
        model_name: The model identifier string from Groq

    Returns:
        ChatGroq LLM instance
    """
    api_key = get_groq_api_key()

    # Use default model if none specified
    if model_name is None:
        models = get_available_models()
        model_name = models["Llama 3.3 70B (Recommended)"]

    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=1024,
        timeout=60,
        max_retries=2,
    )

    return llm


def create_memory() -> ConversationBufferWindowMemory:
    """
    Creates a conversation memory that remembers
    the last 5 question-answer exchanges.

    Returns:
        ConversationBufferWindowMemory instance
    """
    memory = ConversationBufferWindowMemory(
        k=5,                          # Remember last 5 exchanges
        memory_key="chat_history",    # Key used by the chain
        return_messages=True,         # Return as message objects
        output_key="answer"           # Which output to store
    )

    return memory


def build_qa_chain(
    vector_store: FAISS,
    model_name: str = None,
    memory: ConversationBufferWindowMemory = None
) -> ConversationalRetrievalChain:
    """
    Builds the complete conversational QA chain.
    Connects the retriever, LLM, and memory together.

    Args:
        vector_store: FAISS vector store with document embeddings
        model_name: Optional Groq model name to use
        memory: Optional existing memory to continue a conversation

    Returns:
        ConversationalRetrievalChain ready to answer questions
    """
    # Initialize the LLM
    llm = get_llm(model_name)

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Use provided memory or create fresh one
    if memory is None:
        memory = create_memory()

    # Custom prompt to ensure focused, cited answers
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant that answers questions 
based on the provided document context.

Use only the information from the context below to answer the question.
If the answer is not in the context, say "I could not find this information 
in the uploaded documents."

Always be concise, accurate, and helpful.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Build the conversational retrieval chain
    # condense_question_llm is set to the same LLM to avoid
    # a separate call that can cause hanging
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_llm=llm,
        verbose=False,
    )

    return qa_chain


def get_answer(
    qa_chain: ConversationalRetrievalChain,
    question: str
) -> dict:
    """
    Sends a question through the QA chain and returns
    the answer along with source citations.

    Args:
        qa_chain: The built conversational QA chain
        question: User's question string

    Returns:
        Dictionary with 'answer' and 'source_documents'
    """
    response = qa_chain.invoke({"question": question})

    return {
        "answer": response["answer"],
        "source_documents": response["source_documents"]
    }
