from pathlib import Path

from src.document_tracker import (
    needs_processing,
    mark_as_processed,
    get_document_status,
)

from src.ingestion import (
    prepare_single_document
)

from src.vectorstore import (
    add_documents_to_vectorstore,
    update_document,
)


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdf"

MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"


# --------------------------------------------------
# Find documents
# --------------------------------------------------

def get_document_files():

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
# Process one document
# --------------------------------------------------

def process_single_document(
    file_path
):

    file_path = Path(
        file_path
    ).resolve()

    status = get_document_status(
        file_path
    )

    # ----------------------------------------------
    # New document
    # ----------------------------------------------

    if not status["processed"]:

        print(
            "\n" + "-" * 60
        )

        print(
            "🆕 NEW DOCUMENT"
        )

        print(
            file_path.name
        )

        print(
            "-" * 60
        )

        chunks = prepare_single_document(
            file_path
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        add_documents_to_vectorstore(
            chunks
        )

        mark_as_processed(
            file_path
        )

        print(
            f"\n✅ Added: {file_path.name}"
        )

        return "added"

    # ----------------------------------------------
    # Changed document
    # ----------------------------------------------

    if status["changed"]:

        print(
            "\n" + "-" * 60
        )

        print(
            "🔄 CHANGED DOCUMENT"
        )

        print(
            file_path.name
        )

        print(
            "-" * 60
        )

        chunks = prepare_single_document(
            file_path
        )

        print(
            f"Created {len(chunks)} new chunks."
        )

        update_document(
            chunks,
            str(file_path)
        )

        mark_as_processed(
            file_path
        )

        print(
            f"\n✅ Updated: {file_path.name}"
        )

        return "updated"

    # ----------------------------------------------
    # Unchanged
    # ----------------------------------------------

    print(
        f"\n⏭️ SKIPPED: {file_path.name}"
    )

    print(
        "   No changes detected."
    )

    return "skipped"


# --------------------------------------------------
# Process all documents
# --------------------------------------------------

def process_documents():

    print(
        "\n" + "=" * 70
    )

    print(
        "GITGUIDE AI - INCREMENTAL INGESTION"
    )

    print(
        "=" * 70
    )

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

        return

    print(
        f"\n📚 Total documents: {len(files)}"
    )

    added = 0
    updated = 0
    skipped = 0

    # ----------------------------------------------
    # Process every file
    # ----------------------------------------------

    for file_path in files:

        result = process_single_document(
            file_path
        )

        if result == "added":

            added += 1

        elif result == "updated":

            updated += 1

        elif result == "skipped":

            skipped += 1

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "KNOWLEDGE BASE UPDATE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\n🆕 Added:   {added}"
    )

    print(
        f"🔄 Updated: {updated}"
    )

    print(
        f"⏭️ Skipped:  {skipped}"
    )

    print(
        "\nChromaDB is ready."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    process_documents()