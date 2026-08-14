from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.retriever import retrieve_documents


# ============================================================
# OpenAI Client
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# Model
# ============================================================

MODEL = "gpt-4o-mini"


# ============================================================
# Grade Documents
# ============================================================

def grade_documents(question, documents):
    """
    Check whether the retrieved documents contain
    enough information to help answer the user's question.

    Returns:

        True  -> relevant information exists

        False -> relevant information was not found
    """

    if not documents:
        return False

    # --------------------------------------------------------
    # Prepare retrieved content
    # --------------------------------------------------------

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

        if page is not None:
            page = page + 1

        context_parts.append(
            f"""
DOCUMENT {index}

SOURCE:
{source}

PAGE:
{page}

CONTENT:
{document.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Grading prompt
    # --------------------------------------------------------

    prompt = f"""
You are a strict relevance grader for GitGuide AI.

GitGuide AI must answer questions ONLY from the
provided company handbook documents.

Your task is to determine whether the retrieved
documents contain enough information that could
reasonably be used to answer the user's question.

USER QUESTION:
{question}


RETRIEVED DOCUMENTS:
{context}


Important rules:

1. The user's question does NOT need to use the
   same words as the documents.

2. Semantic meaning should be considered.

3. A document is relevant if it contains information
   that can directly or indirectly help answer the
   user's question.

4. Do NOT mark documents as relevant simply because
   they contain a few similar words.

5. If the documents only mention a related topic but
   do not contain information useful for answering
   the question, mark them NOT_RELEVANT.

6. If the question is clearly outside the scope of
   the handbook and the documents cannot answer it,
   mark it NOT_RELEVANT.

7. Do not use outside knowledge.

Return ONLY one of:

RELEVANT

or

NOT_RELEVANT
"""

    # --------------------------------------------------------
    # Ask LLM
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict document relevance "
                    "grader. Follow the provided rules exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0,
    )

    result = (
        response.choices[0]
        .message.content
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # Interpret result
    # --------------------------------------------------------

    if result == "RELEVANT":
        return True

    return False


# ============================================================
# Test Question
# ============================================================

def check_question(question):
    """
    Retrieve documents and check whether the retrieved
    documents are relevant to the question.
    """

    documents = retrieve_documents(
        question,
        k=5
    )

    is_relevant = grade_documents(
        question,
        documents
    )

    return is_relevant, documents


# ============================================================
# Display Test Results
# ============================================================

def display_test_result(
    question,
    relevant,
    documents
):
    """
    Display the grading result and retrieved documents.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "QUESTION"
    )

    print(
        "=" * 70
    )

    print(
        question
    )

    print(
        "\n" + "=" * 70
    )

    if relevant:

        print(
            "RESULT: RELEVANT"
        )

        print(
            "The retrieved documents contain information "
            "that can help answer the question."
        )

    else:

        print(
            "RESULT: NOT RELEVANT"
        )

        print(
            "The retrieved documents do not contain "
            "enough information to answer the question."
        )

    print(
        "=" * 70
    )

    print(
        f"\nRetrieved documents: {len(documents)}"
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

        print(
            f"Source: "
            f"{document.metadata.get('source', 'Unknown')}"
        )

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            print(
                f"Page: {page + 1}"
            )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:500]
        )


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    relevant, documents = check_question(
        question
    )

    display_test_result(
        question,
        relevant,
        documents
    )