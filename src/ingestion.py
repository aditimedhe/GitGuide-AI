from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdf"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"


# --------------------------------------------------
# Load one document
# --------------------------------------------------

def load_single_document(file_path):
    """
    Load one PDF or Markdown document.

    The same loaders used by the original ingestion
    pipeline are used here.
    """

    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    # ----------------------------------------------
    # PDF
    # ----------------------------------------------

    if suffix == ".pdf":

        print(
            f"Loading PDF: {file_path.name}"
        )

        loader = PyPDFLoader(
            str(file_path)
        )

        return loader.load()

    # ----------------------------------------------
    # Markdown
    # ----------------------------------------------

    if suffix in [".md", ".markdown"]:

        print(
            f"Loading Markdown: {file_path.name}"
        )

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        return loader.load()

    # ----------------------------------------------
    # Unsupported file
    # ----------------------------------------------

    raise ValueError(
        f"Unsupported document type: "
        f"{file_path.suffix}"
    )


# --------------------------------------------------
# Load all PDF files
# --------------------------------------------------

def load_pdf_files():
    """
    Load all PDF documents from data/pdf.
    """

    documents = []

    if not PDF_DIR.exists():

        return documents

    for file_path in PDF_DIR.glob("*.pdf"):

        documents.extend(
            load_single_document(
                file_path
            )
        )

    return documents


# --------------------------------------------------
# Load all Markdown files
# --------------------------------------------------

def load_markdown_files():
    """
    Load all Markdown documents from
    data/markdown.
    """

    documents = []

    if not MARKDOWN_DIR.exists():

        return documents

    for file_path in MARKDOWN_DIR.glob("*.md"):

        documents.extend(
            load_single_document(
                file_path
            )
        )

    return documents


# --------------------------------------------------
# Load all documents
# --------------------------------------------------

def load_documents():
    """
    Load all PDF and Markdown documents.
    """

    pdf_documents = load_pdf_files()

    markdown_documents = load_markdown_files()

    documents = (
        pdf_documents +
        markdown_documents
    )

    print(
        f"\nTotal documents loaded: "
        f"{len(documents)}"
    )

    return documents


# --------------------------------------------------
# Split documents
# --------------------------------------------------

def split_documents(documents):
    """
    Split documents into smaller chunks.

    These settings are kept the same as the
    original project.
    """

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

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# --------------------------------------------------
# Prepare one document
# --------------------------------------------------

def prepare_single_document(file_path):
    """
    Load and split one document.

    This function is used by incremental ingestion.
    """

    documents = load_single_document(
        file_path
    )

    if not documents:

        raise ValueError(
            f"No content found in: "
            f"{file_path}"
        )

    chunks = split_documents(
        documents
    )

    return chunks


# --------------------------------------------------
# Prepare all documents
# --------------------------------------------------

def prepare_documents():
    """
    Complete ingestion pipeline for all documents.
    """

    documents = load_documents()

    if not documents:

        raise ValueError(
            "No documents found. "
            "Please place PDFs in data/pdf "
            "or Markdown files in data/markdown."
        )

    chunks = split_documents(
        documents
    )

    return chunks


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    chunks = prepare_documents()

    print(
        "\nSample chunk:"
    )

    print(
        "-" * 60
    )

    print(
        chunks[0].page_content[:1000]
    )

    print(
        "\nMetadata:"
    )

    print(
        chunks[0].metadata
    )