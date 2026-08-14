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
# Query Rewriter
# ============================================================

def rewrite_query(question):
    """
    Rewrite a user's question into a retrieval-friendly
    search query.

    The rewritten query keeps the original meaning while
    extracting important concepts, policies, entities,
    and keywords that may exist in the knowledge base.
    """

    if not question or not question.strip():
        return question

    question = question.strip()

    prompt = f"""
You are a search query optimizer for GitGuide AI.

GitGuide AI searches a company handbook containing
policies, procedures, employee information, programs,
and workplace documentation.

Your job is to transform the user's question into a
better search query for semantic document retrieval.

USER QUESTION:
{question}

Rules:

1. Preserve the original meaning.

2. Do NOT answer the question.

3. Extract the important concepts from the question.

4. Include relevant policy names, topics, entities,
   employee situations, and important keywords when
   they can be inferred from the question.

5. If the question refers to a concept indirectly,
   express that concept explicitly in the search query.

6. Do not add facts that are not implied by the question.

7. Do not make the query unnecessarily long.

8. Return ONLY the improved search query.

Examples:

Question:
"If I'm new to the company, can I take some time off?"

Good search query:
"new employee probation time off leave eligibility"

Question:
"How do promotions work?"

Good search query:
"employee promotion process eligibility criteria"

Question:
"What should I do if I have a problem at work?"

Good search query:
"employee workplace issue reporting support process"

Now rewrite this question:

{question}
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You optimize questions for semantic "
                    "document retrieval. Never answer the question."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0,
    )

    rewritten_query = (
        response.choices[0]
        .message.content
        .strip()
    )

    return rewritten_query


# ============================================================
# Test Query Rewriter
# ============================================================

def display_rewrite(
    original_question,
    rewritten_query
):
    """
    Display the original and rewritten query.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "QUERY REWRITER TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal question:"
    )

    print(
        original_question
    )

    print(
        "\nRewritten search query:"
    )

    print(
        rewritten_query
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    rewritten = rewrite_query(
        question
    )

    display_rewrite(
        question,
        rewritten
    )