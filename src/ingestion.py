from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdf"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"


def load_pdf_files():
    """Load all PDF documents from data/pdf."""

    documents = []

    for file_path in PDF_DIR.glob("*.pdf"):
        print(f"Loading PDF: {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        documents.extend(docs)

    return documents


def load_markdown_files():
    """Load all Markdown documents from data/markdown."""

    documents = []

    for file_path in MARKDOWN_DIR.glob("*.md"):
        print(f"Loading Markdown: {file_path.name}")

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        docs = loader.load()

        documents.extend(docs)

    return documents


def load_documents():
    """Load all PDF and Markdown documents."""

    pdf_documents = load_pdf_files()
    markdown_documents = load_markdown_files()

    documents = pdf_documents + markdown_documents

    print(f"\nTotal documents loaded: {len(documents)}")

    return documents


def split_documents(documents):
    """Split documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def prepare_documents():
    """Complete ingestion pipeline."""

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found. "
            "Please place PDFs in data/pdf "
            "or Markdown files in data/markdown."
        )

    chunks = split_documents(documents)

    return chunks


if __name__ == "__main__":
    chunks = prepare_documents()

    print("\nSample chunk:")
    print("-" * 60)
    print(chunks[0].page_content[:1000])

    print("\nMetadata:")
    print(chunks[0].metadata)