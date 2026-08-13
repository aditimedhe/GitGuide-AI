from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import create_embeddings
from src.ingestion import prepare_documents


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where Chroma will store the database
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Name of our collection
COLLECTION_NAME = "gitguide_handbook"


def create_vectorstore():
    """
    Load documents, create embeddings,
    and store them in ChromaDB.
    """

    print("Preparing documents...")

    chunks = prepare_documents()

    print(f"Number of chunks: {len(chunks)}")

    print("\nCreating embedding model...")

    embeddings = create_embeddings()

    print("Creating Chroma vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("\nVector database created successfully.")
    print(f"Location: {CHROMA_DIR}")

    return vectorstore


if __name__ == "__main__":
    vectorstore = create_vectorstore()

    print("\nTesting similarity search...")

    results = vectorstore.similarity_search(
        "What is the leave policy?",
        k=3,
    )

    print(f"\nRetrieved {len(results)} documents.")

    for index, document in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"Result {index}")
        print("=" * 60)

        print(document.page_content[:500])

        print("\nMetadata:")
        print(document.metadata)