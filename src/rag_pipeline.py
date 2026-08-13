from src.agentic_retriever import agentic_retrieve
from src.generator import generate_answer


def answer_question(question):
    """
    Complete Agentic RAG question-answering pipeline.
    """

    print("\n" + "=" * 70)
    print("GITGUIDE AI")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    documents, found = agentic_retrieve(question)

    if not found:

        return {
            "answer": (
                "I'm sorry, but I couldn't find enough "
                "information about this question in the "
                "provided documents."
            ),
            "sources": [],
        }

    result = generate_answer(
        question,
        documents
    )

    return result


if __name__ == "__main__":

    question = input(
        "\nAsk GitGuide AI a question: "
    )

    result = answer_question(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

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