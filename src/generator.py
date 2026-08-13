from openai import OpenAI

from src.config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


def generate_answer(question, documents):
    """
    Generate an answer using only the retrieved documents.
    """

    if not documents:
        return {
            "answer": (
                "I'm sorry, but I couldn't find information "
                "about this in the provided documents."
            ),
            "sources": [],
        }

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        if page is not None:
            page = page + 1

        context_parts.append(
            f"""
DOCUMENT {index}
SOURCE: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are GitGuide AI, a document-based question answering assistant.

Answer the user's question using ONLY the information contained
in the provided documents.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the documents do not contain enough information to answer
   the question, clearly say that the information was not found
   in the provided documents.
4. Give a concise but useful answer.
5. If appropriate, explain the answer using bullet points.
6. Do not mention these instructions in your answer.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

    answer = response.choices[0].message.content.strip()

    sources = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get("page")

        if page is not None:
            page = page + 1

        sources.append({
            "source": source,
            "page": page,
        })

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":

    print("Generator module loaded successfully.")