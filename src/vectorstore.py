from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import create_embeddings


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "gitguide_handbook"


# --------------------------------------------------
# Create embedding model
# --------------------------------------------------

def get_embedding_model():
    """
    Create the OpenAI embedding model.
    """

    return create_embeddings()


# --------------------------------------------------
# Get ChromaDB
# --------------------------------------------------

def get_vectorstore():
    """
    Load the existing ChromaDB collection.
    """

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


# --------------------------------------------------
# Add documents
# --------------------------------------------------

def add_documents_to_vectorstore(
    documents: List[Document]
):
    """
    Add document chunks to ChromaDB.
    """

    if not documents:

        print(
            "\nNo documents to add."
        )

        return get_vectorstore()

    print(
        f"\nAdding {len(documents)} chunks "
        "to ChromaDB..."
    )

    vectorstore = get_vectorstore()

    vectorstore.add_documents(
        documents
    )

    print(
        "Documents successfully added "
        "to ChromaDB."
    )

    return vectorstore


# --------------------------------------------------
# Delete chunks belonging to one document
# --------------------------------------------------

def delete_documents_by_source(
    source_path: str
):
    """
    Delete all chunks belonging to a specific
    source document.
    """

    vectorstore = get_vectorstore()

    collection = vectorstore._collection

    source_path = str(
        Path(source_path).resolve()
    )

    print(
        f"\nDeleting old chunks for:"
    )

    print(
        source_path
    )

    try:

        collection.delete(
            where={
                "source": source_path
            }
        )

        print(
            "Old chunks deleted successfully."
        )

    except Exception as e:

        print(
            f"Error deleting old chunks: {e}"
        )

        raise

    return vectorstore


# --------------------------------------------------
# Update document
# --------------------------------------------------

def update_document(
    documents: List[Document],
    source_path: str
):
    """
    Replace the chunks belonging to one document.
    """

    print(
        "\nUpdating document:"
    )

    print(
        source_path
    )

    # Delete old chunks
    delete_documents_by_source(
        source_path
    )

    # Add new chunks
    vectorstore = add_documents_to_vectorstore(
        documents
    )

    return vectorstore


# --------------------------------------------------
# Create initial vectorstore
# --------------------------------------------------

def create_vectorstore(
    documents: List[Document] = None
):
    """
    Create or update ChromaDB.

    If documents are supplied, only those documents
    are added.

    If no documents are supplied, all documents are
    loaded using the original ingestion pipeline.
    """

    if documents:

        print(
            f"\nReceived {len(documents)} chunks."
        )

        return add_documents_to_vectorstore(
            documents
        )

    # ----------------------------------------------
    # Backward compatibility
    # ----------------------------------------------

    print(
        "\nPreparing all documents..."
    )

    from src.ingestion import prepare_documents

    chunks = prepare_documents()

    print(
        f"Number of chunks: {len(chunks)}"
    )

    return add_documents_to_vectorstore(
        chunks
    )


# --------------------------------------------------
# Similarity search test
# --------------------------------------------------

if __name__ == "__main__":

    vectorstore = get_vectorstore()

    print(
        "\nTesting similarity search..."
    )

    results = vectorstore.similarity_search(
        "What is the leave policy?",
        k=3,
    )

    print(
        f"\nRetrieved {len(results)} documents."
    )

    for index, document in enumerate(
        results,
        start=1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"Result {index}"
        )

        print(
            "=" * 60
        )

        print(
            document.page_content[:500]
        )

        print(
            "\nMetadata:"
        )

        print(
            document.metadata
        )