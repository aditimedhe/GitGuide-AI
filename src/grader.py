from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.retriever import retrieve_documents


client = OpenAI(api_key=OPENAI_API_KEY)


def grade_documents(question, documents):
    """
    Check whether the retrieved documents are relevant
    to the user's question.
    """

    if not documents:
        return False

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a document relevance grader.

Your job is to determine whether the provided document
content contains information that can help answer the
user's question.

USER QUESTION:
{question}

DOCUMENT CONTENT:
{context}

Return ONLY one word:

RELEVANT

or

NOT_RELEVANT

Do not provide an explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict document relevance grader."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip().upper()

    return result == "RELEVANT"


def check_question(question):
    """
    Retrieve documents and check whether they are relevant.
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


if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    relevant, documents = check_question(question)

    print("\n" + "=" * 60)

    if relevant:
        print("RESULT: RELEVANT")
        print("The documents contain information related to the question.")

    else:
        print("RESULT: NOT RELEVANT")
        print(
            "The provided documents do not contain enough "
            "information to answer this question."
        )

    print("=" * 60)