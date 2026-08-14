from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.retriever import retrieve_context


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL = "gpt-4o-mini"


# --------------------------------------------------
# System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are GitGuide AI, a helpful assistant for
answering questions about the GitGuide knowledge
base.

You must follow these rules:

1. Answer using ONLY the retrieved context.

2. The user does not need to ask the exact
   question that appears in the documents.

3. Use semantic understanding to answer questions
   when the retrieved context contains the
   necessary information.

4. Do not invent facts.

5. Do not use outside knowledge to answer
   GitGuide knowledge-base questions.

6. If the retrieved context does not contain
   enough information to answer the question,
   politely explain that the information is not
   available in the GitGuide knowledge base.

7. Do not pretend that information exists.

8. Keep the answer clear and easy to understand.

9. When possible, mention the source document
   that supports the answer.
"""


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(
    question,
    context
):
    """
    Generate an answer using only retrieved
    document context.
    """

    if not context:

        return (
            "I'm sorry, but I couldn't find relevant "
            "information about this in the GitGuide "
            "knowledge base."
        )

    user_prompt = f"""
Retrieved information from the GitGuide
knowledge base:

{context}


User question:

{question}


Instructions:

Answer the question using only the retrieved
information.

If the retrieved information is not sufficient,
say politely that the information is not available
in the GitGuide knowledge base.
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0,
    )

    return response.choices[0].message.content


# --------------------------------------------------
# Complete RAG Pipeline
# --------------------------------------------------

def ask_question(
    question,
    k=5
):
    """
    Complete basic RAG pipeline.

    Question
        ↓
    ChromaDB retrieval
        ↓
    Context
        ↓
    LLM
        ↓
    Answer
    """

    retrieval_result = retrieve_context(
        question,
        k=k
    )

    documents = retrieval_result[
        "documents"
    ]

    context = retrieval_result[
        "context"
    ]

    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "documents": documents,
        "context": context,
    }


# --------------------------------------------------
# Main Test
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nAsk GitGuide AI: "
    )

    result = ask_question(
        question
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GITGUIDE AI ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 70
    )

    seen_sources = set()

    for document in result["documents"]:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        if source not in seen_sources:

            print(
                source
            )

            seen_sources.add(
                source
            )