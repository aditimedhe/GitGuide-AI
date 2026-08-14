from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.tools import search_handbook
from src.generator import generate_answer


# ============================================================
# OpenAI Client
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# Model
# ============================================================

MODEL = "gpt-4o-mini"


# ============================================================
# STEP 1 — INITIAL QUESTION ROUTING
# ============================================================

def classify_question(question):
    """
    Decide whether the question could potentially
    be answered using the GitGuide handbook.

    This is only an initial routing decision.

    The actual document relevance is checked later
    by agentic_retriever.py.
    """

    prompt = f"""
You are the initial routing component of GitGuide AI.

GitGuide AI answers questions using a company
handbook containing information about topics such as:

- leave and time off
- probation
- onboarding
- career development
- feedback
- employee policies
- workplace guidelines
- people operations
- learning and development
- company programs
- compliance
- employment-related information

User question:

{question}

Determine whether this question could reasonably
be related to information contained in a company
handbook.

Important:

A question does NOT need to use the exact wording
from the handbook.

If the question could potentially be answered using
the handbook, return:

SEARCH

If the question is clearly unrelated to the handbook,
return:

OUT_OF_SCOPE

Return ONLY:

SEARCH

or

OUT_OF_SCOPE
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a routing component for a "
                    "document-based knowledge assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0,
    )

    decision = (
        response.choices[0]
        .message.content
        .strip()
        .upper()
    )

    if "SEARCH" in decision:
        return "SEARCH"

    return "OUT_OF_SCOPE"


# ============================================================
# STEP 2 — RUN GITGUIDE AGENT
# ============================================================

def run_agent(question):
    """
    Run the complete GitGuide AI Agent.

    Workflow:

    User Question
          ↓
    Initial Routing
          ↓
       SEARCH?
       /     \
     NO       YES
     ↓         ↓
    Polite   Agentic Retrieval
    Reply        ↓
              ChromaDB
                ↓
          Relevance Grading
                ↓
          Query Rewriting
                ↓
           Relevant Docs
                ↓
          Answer Generator
                ↓
             Answer
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        return {
            "answer": (
                "Please enter a question about the "
                "GitGuide handbook."
            ),
            "sources": [],
        }

    question = question.strip()

    print("\n" + "=" * 70)
    print("GITGUIDE AI AGENT")
    print("=" * 70)

    print(f"\nUser question:")
    print(question)

    # --------------------------------------------------------
    # Initial routing
    # --------------------------------------------------------

    print("\nStep 1: Classifying question...")

    decision = classify_question(
        question
    )

    print(
        f"Routing decision: {decision}"
    )

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    if decision == "OUT_OF_SCOPE":

        return {
            "answer": (
                "I'm sorry, but this question appears "
                "to be outside the scope of the provided "
                "GitGuide handbook."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # SEARCH HANDBOOK
    # --------------------------------------------------------

    print(
        "\nStep 2: Searching the handbook..."
    )

    search_result = search_handbook(
        question
    )

    # --------------------------------------------------------
    # NO RELEVANT INFORMATION
    # --------------------------------------------------------

    if not search_result["found"]:

        return {
            "answer": (
                "I'm sorry, but I couldn't find enough "
                "relevant information about this in the "
                "provided GitGuide documents."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    print(
        "\nStep 3: Generating grounded answer..."
    )

    result = generate_answer(
        question,
        search_result["documents"],
    )

    return result


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("GITGUIDE AI")
    print("=" * 70)

    question = input(
        "\nAsk GitGuide AI: "
    )

    result = run_agent(
        question
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        result["answer"]
    )

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    if result["sources"]:

        # Avoid displaying duplicate sources
        displayed_sources = set()

        for source in result["sources"]:

            source_name = source.get(
                "source",
                "Unknown source"
            )

            page = source.get(
                "page"
            )

            source_key = (
                source_name,
                page
            )

            if source_key in displayed_sources:
                continue

            displayed_sources.add(
                source_key
            )

            if page is not None:

                print(
                    f"- {source_name} "
                    f"(Page {page})"
                )

            else:

                print(
                    f"- {source_name}"
                )

    else:

        print(
            "No sources available."
        )