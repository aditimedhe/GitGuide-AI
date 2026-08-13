from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.retriever import retrieve_documents
from src.grader import grade_documents
from src.query_rewriter import rewrite_query
from src.generator import generate_answer


MAX_RETRIES = 2


# --------------------------------------------------
# Retrieve
# --------------------------------------------------

def retrieve_node(state: AgentState):

    question = state["question"]

    search_query = state.get(
        "search_query",
        question
    )

    print(f"\n🔎 Searching: {search_query}")

    documents = retrieve_documents(
        search_query,
        k=5
    )

    return {
        "documents": documents
    }


# --------------------------------------------------
# Grade
# --------------------------------------------------

def grade_node(state: AgentState):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    relevant = grade_documents(
        question,
        documents
    )

    print(
        f"📊 Relevance: {relevant}"
    )

    return {
        "relevant": relevant
    }


# --------------------------------------------------
# Rewrite
# --------------------------------------------------

def rewrite_node(state: AgentState):

    question = state["question"]

    current_query = state.get(
        "search_query",
        question
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    retry_count += 1

    print(
        f"🔄 Rewriting query "
        f"(attempt {retry_count})"
    )

    new_query = rewrite_query(
        current_query
    )

    print(
        f"New query: {new_query}"
    )

    return {
        "search_query": new_query,
        "retry_count": retry_count
    }


# --------------------------------------------------
# Generate
# --------------------------------------------------

def generate_node(state: AgentState):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    result = generate_answer(
        question,
        documents
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


# --------------------------------------------------
# Refuse
# --------------------------------------------------

def refuse_node(state: AgentState):

    return {
        "answer": (
            "I'm sorry, but I couldn't find enough "
            "information about this question in the "
            "provided documents."
        ),
        "sources": []
    }


# --------------------------------------------------
# Routing after grading
# --------------------------------------------------

def grade_router(state: AgentState):

    if state.get("relevant", False):

        return "generate"

    retry_count = state.get(
        "retry_count",
        0
    )

    if retry_count < MAX_RETRIES:

        return "rewrite"

    return "refuse"


# --------------------------------------------------
# Build graph
# --------------------------------------------------

def build_graph():

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "grade",
        grade_node
    )

    workflow.add_node(
        "rewrite",
        rewrite_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_node(
        "refuse",
        refuse_node
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "grade"
    )

    workflow.add_conditional_edges(
        "grade",
        grade_router,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "refuse": "refuse"
        }
    )

    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    workflow.add_edge(
        "generate",
        END
    )

    workflow.add_edge(
        "refuse",
        END
    )

    return workflow.compile()


# --------------------------------------------------
# Run graph
# --------------------------------------------------

rag_graph = build_graph()


def ask_agent(question):

    initial_state = {
        "question": question,
        "search_query": question,
        "retry_count": 0
    }

    result = rag_graph.invoke(
        initial_state
    )

    return result


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nAsk GitGuide AI: "
    )

    result = ask_agent(
        question
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        result.get(
            "answer",
            "No answer generated."
        )
    )

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    sources = result.get(
        "sources",
        []
    )

    if sources:

        for source in sources:

            print(
                f"- {source.get('source')} "
                f"(Page {source.get('page')})"
            )

    else:

        print("No sources available.")