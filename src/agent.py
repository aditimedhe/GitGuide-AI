from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.tools import search_handbook
from src.generator import generate_answer


client = OpenAI(api_key=OPENAI_API_KEY)


def run_agent(question):
    """
    Run the GitGuide AI Agent.

    The agent decides whether the user's question
    should be answered using the handbook.
    """

    decision_prompt = f"""
You are GitGuide AI, an assistant that answers questions
using a specific company handbook.

User question:

{question}

Your job is to decide whether this question should be
answered using the company handbook.

Return ONLY one of these:

SEARCH

or

OUT_OF_SCOPE

SEARCH means the question may be answerable from the handbook.

OUT_OF_SCOPE means the question is clearly unrelated
to the handbook.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict routing agent "
                    "for a document-based assistant."
                ),
            },
            {
                "role": "user",
                "content": decision_prompt,
            },
        ],
        temperature=0,
    )

    decision = response.choices[0].message.content.strip().upper()

    if decision == "OUT_OF_SCOPE":

        return {
            "answer": (
                "I'm sorry, but this question is outside "
                "the scope of the provided handbook."
            ),
            "sources": [],
        }

    # Search the knowledge base
    search_result = search_handbook(question)

    if not search_result["found"]:

        return {
            "answer": (
                "I'm sorry, but I couldn't find enough "
                "information about this in the provided "
                "documents."
            ),
            "sources": [],
        }

    # Generate answer only from retrieved documents
    result = generate_answer(
        question,
        search_result["documents"],
    )

    return result


if __name__ == "__main__":

    question = input(
        "\nAsk GitGuide AI: "
    )

    result = run_agent(question)

    print("\n" + "=" * 70)
    print("GITGUIDE AI")
    print("=" * 70)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    if result["sources"]:

        for source in result["sources"]:

            if source["page"] is not None:

                print(
                    f"- {source['source']} "
                    f"(Page {source['page']})"
                )

            else:

                print(
                    f"- {source['source']}"
                )

    else:

        print("No sources available.")