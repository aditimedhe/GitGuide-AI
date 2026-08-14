import streamlit as st

from src.agent import run_agent


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="GitGuide AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📚 GitGuide AI")

    st.markdown(
        """
        ### About

        GitGuide AI is an **Agentic RAG assistant**
        that answers questions using the provided
        company handbook.

        It can:

        - 🔎 Search the knowledge base
        - 🧠 Rewrite difficult queries
        - 📄 Retrieve relevant documents
        - ✅ Check document relevance
        - ✨ Generate grounded answers
        - 🛡️ Validate generated answers
        - 📚 Show document sources
        """
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(
        "GitGuide AI currently uses the documents "
        "stored in the project's knowledge base."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# Main Header
# ============================================================

st.markdown(
    '<div class="main-title">📚 GitGuide AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Your Agentic RAG assistant for the company handbook
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Welcome Message
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Ask me anything about the information "
        "contained in the GitGuide handbook."
    )

    st.markdown("### Example questions")

    example_questions = [
        "What is the leave policy?",
        "What should a new employee know about onboarding?",
        "How does the promotion process work?",
        "What are the rules around time off?",
    ]

    for question in example_questions:

        if st.button(
            question,
            use_container_width=True
        ):

            st.session_state.pending_question = question

            st.rerun()


# ============================================================
# Display Chat History
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------
        # Display sources for assistant messages
        # ----------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    source_name = source.get(
                        "source",
                        "Unknown source"
                    )

                    page = source.get(
                        "page"
                    )

                    if page is not None:

                        st.markdown(
                            f"- `{source_name}` "
                            f"(Page {page})"
                        )

                    else:

                        st.markdown(
                            f"- `{source_name}`"
                        )


# ============================================================
# Get User Question
# ============================================================

pending_question = st.session_state.pop(
    "pending_question",
    None
)

user_question = st.chat_input(
    "Ask a question about the handbook..."
)

question = (
    user_question
    if user_question
    else pending_question
)


# ============================================================
# Process Question
# ============================================================

if question:

    # ----------------------------------------------
    # Display user question
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # ----------------------------------------------
    # Generate answer
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching the handbook..."
        ):

            try:

                result = run_agent(
                    question
                )

                answer = result.get(
                    "answer",
                    "I'm sorry, I couldn't generate an answer."
                )

                sources = result.get(
                    "sources",
                    []
                )

            except Exception as error:

                answer = (
                    "Sorry, something went wrong while "
                    "processing your question."
                )

                sources = []

                st.error(
                    f"Error: {error}"
                )

        # ------------------------------------------
        # Display answer
        # ------------------------------------------

        st.markdown(
            answer
        )

        # ------------------------------------------
        # Display sources
        # ------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                displayed_sources = set()

                for source in sources:

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

                        st.markdown(
                            f"- `{source_name}` "
                            f"(Page {page})"
                        )

                    else:

                        st.markdown(
                            f"- `{source_name}`"
                        )

    # ----------------------------------------------
    # Save assistant message
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )