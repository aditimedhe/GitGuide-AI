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
# Get existing ChromaDB
# --------------------------------------------------

def get_vectorstore():
    """
    Load the existing ChromaDB vector store.

    If the database does not exist yet, it will
    still create/load the collection.
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
    Add new document chunks to ChromaDB.
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
# Delete documents by source
# --------------------------------------------------

def delete_documents_by_source(
    source_path: str
):
    """
    Delete all ChromaDB chunks belonging to
    a particular source document.

    This is required when a document is updated.
    """

    vectorstore = get_vectorstore()

    collection = vectorstore._collection

    print(
        f"\nDeleting old chunks for:"
        f"\n{source_path}"
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
            f"Could not delete old chunks: {e}"
        )

        raise

    return vectorstore


# --------------------------------------------------
# Update one document
# --------------------------------------------------

def update_document(
    documents: List[Document],
    source_path: str
):
    """
    Replace all chunks belonging to one document.

    Steps:

    1. Delete old chunks.
    2. Add new chunks.
    """

    print(
        "\nUpdating document:"
    )

    print(
        source_path
    )

    delete_documents_by_source(
        source_path
    )

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
    Create or update the ChromaDB vector store.

    If documents are supplied, only those documents
    are added.

    If no documents are supplied, this function
    loads documents using the existing ingestion
    pipeline.
    """

    # ----------------------------------------------
    # If documents were supplied
    # ----------------------------------------------

    if documents:

        print(
            f"\nReceived {len(documents)} chunks."
        )

        return add_documents_to_vectorstore(
            documents
        )

    # ----------------------------------------------
    # Backward-compatible initial creation
    # ----------------------------------------------

    print(
        "\nNo documents supplied."
    )

    print(
        "Loading documents using ingestion pipeline..."
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
# Test
# --------------------------------------------------

if __name__ == "__main__":

    vectorstore = create_vectorstore()

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