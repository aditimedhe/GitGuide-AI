import streamlit as st

from src.agent import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GitGuide AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Source box */
    .source-item {
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 8px;
        background-color: #f4f4f5;
        font-size: 14px;
    }

    /* Status box */
    .status-box {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #f4f4f5;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 GitGuide AI")

    st.caption(
        "Agentic RAG Company Handbook Assistant"
    )

    st.divider()

    st.subheader("About GitGuide AI")

    st.write(
        """
        GitGuide AI uses an Agentic RAG pipeline to
        answer questions from the provided company
        handbook.
        """
    )

    st.markdown(
        """
        **Pipeline**

        🔎 Query Routing  
        ↓  
        📚 Document Retrieval  
        ↓  
        🧠 Relevance Grading  
        ↓  
        ✍️ Query Rewriting  
        ↓  
        📖 Grounded Answer  
        ↓  
        ✅ Answer Validation
        """
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(
        "GitGuide AI searches the documents stored "
        "in the project knowledge base."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 GitGuide AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the company handbook and
    get answers grounded in the provided documents.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome to GitGuide AI! "
        "Ask a question about the company handbook."
    )

    st.markdown(
        "### 💡 Try asking"
    )

    example_questions = [
        "What is the leave policy?",
        "What should I know about onboarding?",
        "How does the promotion process work?",
        "What are the rules around time off?",
    ]

    columns = st.columns(2)

    for index, example in enumerate(
        example_questions
    ):

        with columns[index % 2]:

            if st.button(
                example,
                use_container_width=True
            ):

                st.session_state.pending_question = example

                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 View Sources"
            ):

                displayed_sources = set()

                for source in message["sources"]:

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
                            f"""
                            <div class="source-item">
                            📄 <b>{source_name}</b>
                            — Page {page}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="source-item">
                            📄 <b>{source_name}</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ============================================================
# USER INPUT
# ============================================================

pending_question = st.session_state.pop(
    "pending_question",
    None
)

user_question = st.chat_input(
    "Ask something about the handbook..."
)

question = (
    user_question
    if user_question
    else pending_question
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_placeholder = st.empty()

        status_placeholder.markdown(
            """
            <div class="status-box">
            🔎 Searching the GitGuide knowledge base...
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner(
            "GitGuide AI is thinking..."
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

                status_placeholder.empty()

            except Exception as error:

                status_placeholder.empty()

                answer = (
                    "I'm sorry, something went wrong "
                    "while processing your question."
                )

                sources = []

                st.error(
                    f"Technical error: {error}"
                )

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 View Sources"
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
                            f"- 📄 `{source_name}` "
                            f"(Page {page})"
                        )

                    else:

                        st.markdown(
                            f"- 📄 `{source_name}`"
                        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )