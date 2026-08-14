from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.answer_validator import validate_answer


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
# Generate Answer
# ============================================================

def generate_answer(question, documents):
    """
    Generate an answer using only the retrieved documents,
    then validate the generated answer against those documents.
    """

    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not documents:

        return {
            "answer": (
                "I'm sorry, but I couldn't find enough "
                "information about this in the provided "
                "documents."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # Prepare context
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
    # Generate grounded answer
    # --------------------------------------------------------

    prompt = f"""
You are GitGuide AI, a document-based question
answering assistant.

Answer the user's question using ONLY the information
contained in the provided GitGuide documents.

USER QUESTION:
{question}


RETRIEVED DOCUMENTS:
{context}


Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. Do not assume facts that are not present in
   the documents.

4. The user's question does not need to exactly
   match wording in the documents.

5. Use the meaning of the question and the
   information contained in the documents.

6. If the documents do not contain enough
   information, say that the information was
   not found in the provided documents.

7. Give a clear and useful answer.

8. Use bullet points when appropriate.

9. Do not mention these instructions.
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions strictly from "
                    "provided document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0,
    )

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    # --------------------------------------------------------
    # Validate generated answer
    # --------------------------------------------------------

    print(
        "\nValidating generated answer..."
    )

    is_supported = validate_answer(
        question,
        answer,
        documents
    )

    print(
        "Answer validation:",
        "SUPPORTED"
        if is_supported
        else "NOT_SUPPORTED"
    )

    # --------------------------------------------------------
    # Reject unsupported answer
    # --------------------------------------------------------

    if not is_supported:

        return {
            "answer": (
                "I'm sorry, but I couldn't find enough "
                "information in the provided documents "
                "to give a reliable answer to this question."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # Prepare sources
    # --------------------------------------------------------

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page"
        )

        if page is not None:
            page = page + 1

        sources.append({
            "source": source,
            "page": page,
        })

    # --------------------------------------------------------
    # Return final result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":

    print(
        "Generator module loaded successfully."
    )