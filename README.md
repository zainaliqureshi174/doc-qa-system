# 📄 Document Q&A System

A production-ready Document Q&A System built with LangChain, FAISS, and Streamlit. Upload your documents and ask questions in natural language — the system finds accurate, context-aware answers and shows you exactly where they came from.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.10.0-orange.svg)

---

## ✨ Features

- **Multi-Format Support** — Upload PDF, DOCX, and TXT files
- **Semantic Search** — Finds answers by meaning, not just keywords
- **Conversational Memory** — Ask follow-up questions naturally
- **Source Citations** — See exactly which part of the document answered your question
- **Multi-Document Support** — Upload multiple documents and query across all of them
- **Document Summarization** — Generate a full document summary in one click
- **Model Flexibility** — Switch between Groq LLM models from the UI
- **Chat History** — Full conversation history displayed in a clean chat interface

---

## 🏗️ Architecture

The system works in two phases:

**Phase 1 — Ingestion (when a document is uploaded):**
```
Document → Parse Text → Split into Chunks → Generate Embeddings → Store in FAISS
```

**Phase 2 — Query (when a question is asked):**
```
Question → Embed Question → Search FAISS → Retrieve Relevant Chunks → LLM generates Answer
```

Conversational memory passes previous Q&A pairs into the LLM context so follow-up questions work naturally.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain 0.3.25 |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS |
| UI | Streamlit |
| Document Loaders | PyPDF, Docx2txt, TextLoader |
| Memory | ConversationBufferWindowMemory |

---

## 📁 Project Structure
```
doc_qa_system/
│
├── app.py                        # Streamlit entry point
├── .env                          # API keys (not committed)
├── .env.example                  # Environment variable template
├── requirements.txt              # All dependencies
├── README.md                     # Project documentation
│
├── core/
│   ├── document_processor.py     # Document loading and chunking
│   ├── vector_store.py           # FAISS embeddings and retrieval
│   ├── qa_chain.py               # LangChain conversational chain
│   └── summarizer.py             # Map-reduce summarization
│
└── utils/
    └── helpers.py                # Configuration and utilities
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/zainaliqureshi174/doc-qa-system.git
cd doc-qa-system
```

**2. Create and activate virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the application:**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 💡 How to Use

1. **Upload a document** — Use the sidebar to upload a PDF, DOCX, or TXT file
2. **Ask questions** — Type your question in the chat input
3. **View sources** — Click "View Sources" under any answer to see citations
4. **Follow-up questions** — Ask follow-up questions naturally using "he", "it", "they"
5. **Summarize** — Click "Generate Summary" for a full document overview
6. **Multiple documents** — Upload more files to query across all of them
7. **Clear chat** — Use "Clear Chat" to start a new conversation
8. **Reset** — Use "Reset All" to clear everything and start fresh

---

## 🔧 Configuration

### Switching LLM Models
Select from available models in the sidebar:
- **Llama 3.3 70B** — Best quality, recommended
- **Llama 3.1 8B** — Faster responses
- **Mixtral 8x7B** — Alternative option

### Chunking Parameters
Adjust in `core/document_processor.py`:
```python
chunk_size=1000      # Characters per chunk
chunk_overlap=200    # Overlap between chunks
```

### Retrieval Settings
Adjust in `core/qa_chain.py`:
```python
search_kwargs={"k": 4}    # Number of chunks retrieved per query
```

---

## 📝 Environment Variables

| Variable | Description | Required |
|---|---|---|
| GROQ_API_KEY | Your Groq API key | Yes |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with LangChain Chains — no agents required.*
