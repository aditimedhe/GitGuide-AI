from openai import OpenAI

from src.config import OPENAI_API_KEY


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
# Validate Answer
# ============================================================

def validate_answer(question, answer, documents):
    """
    Check whether the generated answer is supported
    by the retrieved documents.

    Returns:

        True  -> answer is supported

        False -> answer is not sufficiently supported
    """

    if not answer or not documents:
        return False

    # --------------------------------------------------------
    # Build document context
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
    # Validation prompt
    # --------------------------------------------------------

    prompt = f"""
You are the final answer validator for GitGuide AI.

Your job is to determine whether the generated answer
is fully supported by the retrieved company handbook
documents.

USER QUESTION:
{question}


GENERATED ANSWER:
{answer}


RETRIEVED DOCUMENTS:
{context}


Rules:

1. The answer must be supported by the documents.

2. Do not use outside knowledge when judging support.

3. The answer does not need to use the exact wording
   of the documents.

4. Reasonable paraphrasing is allowed.

5. If the answer contains information that cannot be
   supported by the documents, mark it NOT_SUPPORTED.

6. If the documents do not provide enough information
   to answer the question, mark it NOT_SUPPORTED.

7. If the answer is completely supported by the
   retrieved documents, mark it SUPPORTED.

Return ONLY one of:

SUPPORTED

or

NOT_SUPPORTED
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
                    "You are a strict factual grounding "
                    "validator for a RAG system."
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

    return result == "SUPPORTED"


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print(
        "\nAnswer validator module loaded successfully."
    )