from langchain_openai import OpenAIEmbeddings

from src.config import OPENAI_API_KEY


def create_embeddings():
    """
    Create the OpenAI embedding model used
    to convert text into vectors.
    """

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_embeddings()

    test_text = "What is the leave policy?"

    vector = embeddings.embed_query(test_text)

    print("Embedding created successfully.")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")