from src.retriever import retrieve_documents
from src.grader import grade_documents
from src.query_rewriter import rewrite_query


# ============================================================
# Settings
# ============================================================

MAX_RETRIES = 2
TOP_K = 5


# ============================================================
# Agentic Retrieval
# ============================================================

def agentic_retrieve(question):
    """
    Perform iterative Agentic RAG retrieval.

    Workflow:

        Original Question
                ↓
          Retrieve Documents
                ↓
          Grade Documents
             /       \
        Relevant   Not Relevant
           ↓           ↓
        Return      Rewrite Query
                        ↓
                    Retrieve Again
                        ↓
                      Grade
                        ↓
                  Maximum retries
                        ↓
                  Return / Reject
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        print("Empty question received.")

        return [], False

    original_question = question.strip()

    current_query = original_question

    # --------------------------------------------------------
    # Retrieval loop
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES + 1):

        print("\n" + "=" * 70)

        print(
            f"RETRIEVAL ATTEMPT: {attempt + 1}"
        )

        print("=" * 70)

        print(
            f"\nOriginal question:\n{original_question}"
        )

        print(
            f"\nCurrent search query:\n{current_query}"
        )

        # ----------------------------------------------------
        # Retrieve documents
        # ----------------------------------------------------

        print(
            "\nSearching ChromaDB..."
        )

        documents = retrieve_documents(
            current_query,
            k=TOP_K
        )

        print(
            f"Retrieved documents: {len(documents)}"
        )

        # ----------------------------------------------------
        # No documents
        # ----------------------------------------------------

        if not documents:

            print(
                "No documents retrieved."
            )

            if attempt < MAX_RETRIES:

                print(
                    "\nTrying a rewritten query..."
                )

                current_query = rewrite_query(
                    original_question
                )

                print(
                    f"Rewritten query:\n{current_query}"
                )

                continue

            print(
                "\nMaximum retrieval attempts reached."
            )

            return [], False

        # ----------------------------------------------------
        # Grade documents
        # ----------------------------------------------------

        print(
            "\nChecking document relevance..."
        )

        is_relevant = grade_documents(
            original_question,
            documents
        )

        print(
            f"Relevance result: "
            f"{'RELEVANT' if is_relevant else 'NOT_RELEVANT'}"
        )

        # ----------------------------------------------------
        # Relevant documents found
        # ----------------------------------------------------

        if is_relevant:

            print(
                "\nRelevant documents found."
            )

            print(
                "Stopping retrieval."
            )

            return documents, True

        # ----------------------------------------------------
        # Not relevant — retry
        # ----------------------------------------------------

        if attempt < MAX_RETRIES:

            print(
                "\nRetrieved documents were not relevant."
            )

            print(
                "Rewriting the original question..."
            )

            current_query = rewrite_query(
                original_question
            )

            print(
                f"\nNew search query:\n{current_query}"
            )

        else:

            print(
                "\nMaximum retrieval attempts reached."
            )

            print(
                "No relevant information found."
            )

    return [], False


# ============================================================
# Display Results
# ============================================================

def display_results(
    question,
    documents,
    found
):
    """
    Display retrieval results for testing.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "AGENTIC RETRIEVAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestion:\n{question}"
    )

    if not found:

        print(
            "\nRESULT: NO RELEVANT INFORMATION"
        )

        print(
            "\nGitGuide could not find enough "
            "relevant information in the knowledge base."
        )

        return

    print(
        f"\nRESULT: RELEVANT INFORMATION FOUND"
    )

    print(
        f"Documents returned: {len(documents)}"
    )

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"DOCUMENT {index}"
        )

        print(
            "-" * 70
        )

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page"
        )

        print(
            f"Source: {source}"
        )

        if page is not None:

            print(
                f"Page: {page + 1}"
            )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:700]
        )


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    documents, found = agentic_retrieve(
        question
    )

    display_results(
        question,
        documents,
        found
    )