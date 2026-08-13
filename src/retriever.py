from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import create_embeddings


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chroma database location
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Same collection name used when creating Chroma
COLLECTION_NAME = "gitguide_handbook"


def get_vectorstore():
    """
    Connect to the existing Chroma vector database.
    """

    embeddings = create_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


def retrieve_documents(question, k=5):
    """
    Retrieve the most relevant document chunks
    for a user's question.
    """

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k,
    )

    return results


def display_results(question):
    """
    Test retrieval and display the results.
    """

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    results = retrieve_documents(question)

    if not results:
        print("No documents were retrieved.")
        return

    for index, document in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print("-" * 70)

        print(document.page_content[:800])

        print("\nSOURCE METADATA:")
        print(document.metadata)


if __name__ == "__main__":

    test_question = input(
        "\nEnter your question about the handbook: "
    )

    display_results(test_question)