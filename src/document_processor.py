from pathlib import Path

from src.document_tracker import (
    needs_processing,
    mark_as_processed,
    get_document_status,
)

from src.vectorstore import create_vectorstore


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdf"

MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"


# --------------------------------------------------
# Get documents
# --------------------------------------------------

def get_document_files():
    """
    Find all PDF and Markdown documents.
    """

    files = []

    if PDF_DIR.exists():

        files.extend(
            PDF_DIR.glob("*.pdf")
        )

    if MARKDOWN_DIR.exists():

        files.extend(
            MARKDOWN_DIR.glob("*.md")
        )

        files.extend(
            MARKDOWN_DIR.glob("*.markdown")
        )

    return sorted(files)


# --------------------------------------------------
# Find new or changed documents
# --------------------------------------------------

def get_documents_to_process():
    """
    Return only documents that are new or changed.
    """

    files = get_document_files()

    documents_to_process = []

    for file in files:

        if needs_processing(file):

            documents_to_process.append(file)

    return documents_to_process


# --------------------------------------------------
# Show status
# --------------------------------------------------

def show_document_status():
    """
    Display processing status of every document.
    """

    files = get_document_files()

    print("\n" + "=" * 70)
    print("DOCUMENT STATUS")
    print("=" * 70)

    if not files:

        print(
            "\nNo documents found."
        )

        return

    for file in files:

        status = get_document_status(
            file
        )

        if not status["processed"]:

            label = "🆕 NEW"

        elif status["changed"]:

            label = "🔄 CHANGED"

        else:

            label = "✅ PROCESSED"

        print(
            f"{label}  {file.name}"
        )


# --------------------------------------------------
# Process documents
# --------------------------------------------------

def process_documents():
    """
    Detect new/changed documents and process
    the knowledge base.
    """

    print("\n" + "=" * 70)
    print("GITGUIDE AI - INCREMENTAL DOCUMENT PROCESSOR")
    print("=" * 70)

    files = get_document_files()

    if not files:

        print(
            "\n❌ No PDF or Markdown documents found."
        )

        print(
            f"\nPDF folder: {PDF_DIR}"
        )

        print(
            f"Markdown folder: {MARKDOWN_DIR}"
        )

        return None

    print(
        f"\n📚 Total documents found: {len(files)}"
    )

    # ----------------------------------------------
    # Determine what needs processing
    # ----------------------------------------------

    documents_to_process = (
        get_documents_to_process()
    )

    print(
        f"\n🆕 New/changed documents: "
        f"{len(documents_to_process)}"
    )

    # ----------------------------------------------
    # Nothing to process
    # ----------------------------------------------

    if not documents_to_process:

        print(
            "\n✅ Knowledge base is already up to date."
        )

        print(
            "\nNo embeddings need to be created."
        )

        return None

    # ----------------------------------------------
    # Display documents
    # ----------------------------------------------

    print(
        "\nDocuments that require processing:"
    )

    for file in documents_to_process:

        print(
            f"   📄 {file.name}"
        )

    # ----------------------------------------------
    # Current limitation
    # ----------------------------------------------

    print(
        "\n⚠️ Processing new documents..."
    )

    print(
        "The current vectorstore implementation "
        "rebuilds the collection."
    )

    print(
        "Incremental Chroma insertion will be "
        "implemented in the next step."
    )

    # ----------------------------------------------
    # Run existing vectorstore pipeline
    # ----------------------------------------------

    try:

        vectorstore = create_vectorstore()

    except Exception as e:

        print(
            "\n❌ Vector database update failed."
        )

        print(
            f"\nError: {e}"
        )

        raise

    # ----------------------------------------------
    # Mark documents as processed
    # ----------------------------------------------

    for file in documents_to_process:

        mark_as_processed(
            file
        )

    # ----------------------------------------------
    # Success
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print(
        "✅ DOCUMENT PROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        "\nProcessed documents:"
    )

    for file in documents_to_process:

        print(
            f"   ✅ {file.name}"
        )

    return vectorstore


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    show_document_status()

    process_documents()