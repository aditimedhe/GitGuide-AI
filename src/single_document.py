from pathlib import Path

from src.ingestion import load_single_document


def prepare_single_document(
    file_path
):
    """
    Load and prepare one PDF or Markdown document.
    """

    file_path = Path(file_path)

    print(
        f"\nLoading document:"
    )

    print(
        file_path.name
    )

    documents = load_single_document(
        file_path
    )

    print(
        f"Loaded {len(documents)} document sections."
    )

    return documents