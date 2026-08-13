from openai import OpenAI

from src.config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


def rewrite_query(question):
    """
    Rewrite the user's question to improve document retrieval.
    """

    prompt = f"""
You are a search query optimizer for a document-based
question answering system.

Rewrite the user's question so that it is easier to search
within the provided knowledge base.

Keep the original meaning.

Do not answer the question.

Return ONLY the rewritten search query.

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You optimize questions for document retrieval."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )

    rewritten_query = response.choices[0].message.content.strip()

    return rewritten_query


if __name__ == "__main__":

    question = input("\nEnter your question: ")

    rewritten = rewrite_query(question)

    print("\nOriginal question:")
    print(question)

    print("\nRewritten search query:")
    print(rewritten)