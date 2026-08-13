from src.retriever import retrieve_documents
from src.grader import grade_documents
from src.query_rewriter import rewrite_query


MAX_RETRIES = 2


def agentic_retrieve(question):
    """
    Retrieve relevant documents using an iterative
    retrieval and query-rewriting process.
    """

    current_query = question

    for attempt in range(MAX_RETRIES + 1):

        print("\n" + "=" * 70)
        print(f"RETRIEVAL ATTEMPT: {attempt + 1}")
        print("=" * 70)

        print(f"Search query: {current_query}")

        documents = retrieve_documents(
            current_query,
            k=5
        )

        print(f"Retrieved documents: {len(documents)}")

        if not documents:
            print("No documents found.")
            return [], False

        is_relevant = grade_documents(
            question,
            documents
        )

        print(f"Relevance result: {is_relevant}")

        if is_relevant:
            print("Relevant documents found.")
            return documents, True

        if attempt < MAX_RETRIES:

            print("\nRetrieved content was not relevant.")
            print("Rewriting search query...")

            current_query = rewrite_query(
                current_query
            )

            print(
                f"New search query: {current_query}"
            )

        else:
            print("\nMaximum retrieval attempts reached.")
            print("No relevant information found.")

    return [], False


if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    documents, found = agentic_retrieve(question)

    print("\n" + "=" * 70)

    if found:

        print("SUCCESS: Relevant information found.")

        for index, document in enumerate(
            documents,
            start=1
        ):

            print("\n" + "-" * 70)
            print(f"DOCUMENT {index}")
            print("-" * 70)

            print(
                document.page_content[:500]
            )

            print("\nMetadata:")
            print(document.metadata)

    else:

        print(
            "NO RELEVANT INFORMATION FOUND "
            "IN THE KNOWLEDGE BASE."
        )