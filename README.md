# 📚 GitGuide AI

**GitGuide AI** is an **Agentic RAG (Retrieval-Augmented Generation)** assistant that answers questions about a company handbook — built and tested on GitLab's publicly available People Handbook (leave policy, onboarding, promotions, feedback, compliance, and more).

Unlike a basic RAG chatbot that does a single vector search and generates an answer, GitGuide AI runs a **self-correcting agentic loop**: it retrieves documents, grades their relevance, rewrites the search query if the results aren't good enough, retries, and finally validates that its own answer is actually grounded in the retrieved sources before showing it to the user. If it can't find or verify a supported answer, it says so instead of guessing.

---

## ✨ Key Features

- **Agentic retrieval loop** — retrieves → grades relevance → rewrites the query → retries (up to 2 times) until relevant context is found, or gracefully gives up.
- **LLM-based relevance grading** — an LLM checks whether retrieved chunks can *actually* answer the question (semantic, not keyword matching).
- **Query rewriting** — turns vague or conversational questions into strong, keyword-rich search queries for the vector store.
- **Grounded answer generation** — answers are generated strictly from retrieved context, never from the model's general knowledge.
- **Answer validation (hallucination check)** — a second LLM pass verifies the generated answer is fully supported by the retrieved documents before it's returned; unsupported answers are rejected and replaced with an honest "not found" response.
- **Out-of-scope routing** — an initial classifier filters out questions that clearly have nothing to do with the handbook before doing any retrieval.
- **Incremental document ingestion** — a SHA-256 based document tracker detects new or changed files so only new/updated documents are re-embedded, not the whole corpus.
- **Source citations** — every answer comes with the originating document name (and page number for PDFs) shown in an expandable "View Sources" panel.
- **Streamlit chat UI** — clean chat interface with example prompts, conversation history, and a sidebar explaining the pipeline.
- **Also runnable via LangGraph** — the same agentic flow is implemented both as hand-rolled Python control flow (`agentic_retriever.py`) and as a `LangGraph` `StateGraph` (`graph.py`), for experimentation/reference.

---

## 🧠 How It Works (Pipeline)

```
 User Question
      │
      ▼
 1. Initial Routing (LLM classifier)
      │
   SEARCH? ──── NO ──► Polite "out of scope" reply
      │
     YES
      │
      ▼
 2. Agentic Retrieval
      │
      ├─► Retrieve top-k chunks from ChromaDB (semantic search)
      │
      ├─► Grade relevance (LLM judges: RELEVANT / NOT_RELEVANT)
      │
      ├─► If NOT relevant and retries remain:
      │        Rewrite query (LLM) ──► retry retrieval
      │
      └─► If still not relevant after max retries:
               Return "no relevant information found"
      │
      ▼
 3. Grounded Answer Generation (LLM, context-only)
      │
      ▼
 4. Answer Validation (LLM checks answer is SUPPORTED by sources)
      │
   SUPPORTED? ── NO ──► Reject, return honest "couldn't verify" message
      │
     YES
      │
      ▼
 Final Answer + Cited Sources
```

---

## 🗂️ Project Structure

```
GitGuide-AI/
├── app.py                      # Streamlit chat UI (entry point)
├── requirements.txt
├── .env                        # OPENAI_API_KEY (not committed)
├── data/
│   ├── pdf/                    # Source handbook documents (PDF)
│   └── markdown/                # Source handbook documents (Markdown)
├── vectorstore/                 # Persisted ChromaDB collection + tracker
│   └── processed_documents.json
└── src/
    ├── config.py                # Loads OPENAI_API_KEY from .env
    ├── state.py                 # LangGraph AgentState schema
    │
    ├── ingestion.py              # Loads PDFs/Markdown, chunks documents
    ├── single_document.py        # Load + prepare a single document
    ├── document_tracker.py       # SHA-256 hashing to detect new/changed files
    ├── document_processor.py     # Orchestrates ingest → embed → track per file
    ├── embeddings.py             # OpenAI text-embedding-3-small wrapper
    ├── vectorstore.py            # ChromaDB add / update / delete operations
    │
    ├── retriever.py              # Similarity search against ChromaDB
    ├── grader.py                 # LLM relevance grading of retrieved docs
    ├── query_rewriter.py         # LLM query rewriting for better retrieval
    ├── agentic_retriever.py      # Retrieve → grade → rewrite → retry loop
    │
    ├── generator.py              # Grounded answer generation + validation call
    ├── answer_generator.py       # Simpler standalone RAG answer generator
    ├── answer_validator.py       # LLM groundedness / hallucination check
    │
    ├── agent.py                  # Full agent: routing → retrieval → answer
    ├── tools.py                  # search_handbook() tool wrapper
    ├── rag_pipeline.py           # End-to-end pipeline entry point
    └── graph.py                  # LangGraph StateGraph implementation of the loop
```

---

## 🛠️ Tech Stack

| Layer              | Technology                                                    |
|---------------------|----------------------------------------------------------------|
| LLM                 | OpenAI `gpt-4o-mini`                                           |
| Embeddings          | OpenAI `text-embedding-3-small`                                |
| Vector store        | ChromaDB (`langchain-chroma`)                                  |
| Orchestration       | LangChain + LangGraph                                          |
| Document loading    | `PyPDFLoader`, `TextLoader`, `RecursiveCharacterTextSplitter`  |
| UI                  | Streamlit                                                       |
| Change detection    | SHA-256 file hashing (custom document tracker)                 |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd GitGuide-AI
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Add your documents

Place your handbook files in:

```
data/pdf/        # .pdf files
data/markdown/   # .md files
```

### 5. Ingest documents into the vector store

Run the document processor to chunk, embed, and store your documents in ChromaDB (only new/changed files are processed thanks to the document tracker):

```bash
python -m src.document_processor
```

### 6. Launch the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) and start asking questions.

---

## 💬 Example Questions

- "What is the leave policy?"
- "What should I know about onboarding?"
- "How does the promotion process work?"
- "What are the rules around time off?"
- "If I'm new to the company, can I take some time off?"

---

## 🧪 Testing Individual Components

Most modules can be run directly for isolated testing:

```bash
python -m src.retriever          # Test raw vector similarity search
python -m src.grader             # Test relevance grading on a question
python -m src.query_rewriter     # Test query rewriting
python -m src.agentic_retriever  # Test the full retrieve→grade→rewrite loop
python -m src.agent              # Test the full agent end-to-end (CLI)
python -m src.graph              # Test the LangGraph implementation
```

---

## 🔒 Grounding & Anti-Hallucination Design

GitGuide AI is intentionally conservative:

- It **never** answers from the model's general/world knowledge — only from retrieved handbook content.
- Every answer passes through a **separate validation pass** that checks whether the claims in the answer are actually supported by the retrieved documents.
- If retrieval fails, grading fails, or validation fails at any stage, the assistant returns a clear "I couldn't find/verify this in the provided documents" response rather than guessing.

---

## 📌 Notes

- The included example knowledge base is built from GitLab's publicly available People Handbook, used here purely as sample data to demonstrate the agentic RAG pipeline. Swap in your own `data/pdf` and `data/markdown` files to point GitGuide AI at a different handbook or knowledge base.
- `vectorstore/` and `venv/` are excluded from version control via `.gitignore` — regenerate the vector store locally by running the ingestion step above.
- Do not commit your `.env` file; `OPENAI_API_KEY` must be supplied locally or via your deployment platform's secret manager.

---

## 📄 License

Add your preferred license here (e.g. MIT).
