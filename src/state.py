from typing import TypedDict, List, Any


class AgentState(TypedDict, total=False):

    question: str

    search_query: str

    documents: List[Any]

    relevant: bool

    retry_count: int

    answer: str

    sources: List[dict]