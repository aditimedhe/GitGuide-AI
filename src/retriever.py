from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import create_embeddings


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chroma database location
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Same collection name used when creating Chroma
COLLECTION_NAME = "gitguide_handbook"


# --------------------------------------------------
# Get Vector Database
# --------------------------------------------------

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


# --------------------------------------------------
# Retrieve Documents
# --------------------------------------------------

def retrieve_documents(question, k=5):
    """
    Retrieve the most relevant document chunks
    for a user's question.
    """

    if not question or not question.strip():
        return []

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k,
    )

    return results


# --------------------------------------------------
# Retrieve Documents With Scores
# --------------------------------------------------

def retrieve_documents_with_scores(
    question,
    k=5
):
    """
    Retrieve relevant document chunks along
    with their similarity distance scores.

    Lower distance generally means the result
    is more similar to the question.
    """

    if not question or not question.strip():
        return []

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    return results


# --------------------------------------------------
# Format Context
# --------------------------------------------------

def format_context(documents):
    """
    Convert retrieved documents into a context
    string that can later be passed to the LLM.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page"
        )

        source_information = source

        # PDF pages are normally zero-indexed
        if page is not None:

            source_information += (
                f" | Page {page + 1}"
            )

        context_parts.append(
            f"""
--- CONTEXT {index} ---

Source:
{source_information}

Content:
{document.page_content}
"""
        )

    return "\n".join(
        context_parts
    )


# --------------------------------------------------
# Retrieve Context
# --------------------------------------------------

def retrieve_context(
    question,
    k=5
):
    """
    Retrieve documents and prepare them as
    context for the answer-generation step.
    """

    documents = retrieve_documents(
        question,
        k=k
    )

    context = format_context(
        documents
    )

    return {
        "documents": documents,
        "context": context,
    }


# --------------------------------------------------
# Display Results
# --------------------------------------------------

def display_results(question):
    """
    Test retrieval and display the results.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        "=" * 70
    )

    results = retrieve_documents(
        question
    )

    if not results:

        print(
            "No documents were retrieved."
        )

        return

    for index, document in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            "-" * 70
        )

        print(
            document.page_content[:800]
        )

        print(
            "\nSOURCE METADATA:"
        )

        print(
            document.metadata
        )


# --------------------------------------------------
# Test Similarity Scores
# --------------------------------------------------

def display_results_with_scores(
    question
):
    """
    Display retrieved documents together
    with their similarity distance scores.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        "=" * 70
    )

    results = retrieve_documents_with_scores(
        question
    )

    if not results:

        print(
            "No documents were retrieved."
        )

        return

    for index, (
        document,
        score
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            "-" * 70
        )

        print(
            f"Distance Score: {score}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source')}"
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:800]
        )


# --------------------------------------------------
# Main Test
# --------------------------------------------------

if __name__ == "__main__":

    test_question = input(
        "\nEnter your question about the handbook: "
    )

    display_results_with_scores(
        test_question
    )