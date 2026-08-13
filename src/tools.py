from src.agentic_retriever import agentic_retrieve


def search_handbook(question):
    """
    Search the GitGuide handbook and return relevant documents.
    """

    documents, found = agentic_retrieve(question)

    if not found:
        return {
            "found": False,
            "documents": [],
            "message": (
                "No relevant information was found "
                "in the provided handbook."
            ),
        }

    return {
        "found": True,
        "documents": documents,
        "message": "Relevant handbook information found.",
    }