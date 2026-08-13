import streamlit as st

from src.graph import ask_agent


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="GitGuide AI",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 GitGuide AI")

st.subheader("Agentic RAG Knowledge Assistant")

st.write(
    "Ask questions about the documents in the GitGuide "
    "knowledge base."
)

st.divider()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📚 GitGuide AI")

    st.write(
        "This assistant answers questions using "
        "the provided knowledge base."
    )

    st.divider()

    st.subheader("Features")

    st.write("✅ Document-based answers")
    st.write("✅ Semantic search")
    st.write("✅ Agentic retrieval")
    st.write("✅ Query rewriting")
    st.write("✅ Relevance checking")
    st.write("✅ Source references")
    st.write("✅ Out-of-document detection")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    source_name = source.get(
                        "source",
                        "Unknown source"
                    )

                    page = source.get("page")

                    if page is not None:

                        st.write(
                            f"📄 {source_name} — "
                            f"Page {page}"
                        )

                    else:

                        st.write(
                            f"📄 {source_name}"
                        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the documents..."
)


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
        st.markdown(question)


    # ----------------------------------------------
    # Run Agent
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching the knowledge base..."
        ):

            try:

                result = ask_agent(question)

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer."
                )

                sources = result.get(
                    "sources",
                    []
                )

                st.markdown(answer)


                # ----------------------------------
                # Sources
                # ----------------------------------

                if sources:

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for source in sources:

                            source_name = source.get(
                                "source",
                                "Unknown source"
                            )

                            page = source.get(
                                "page"
                            )

                            if page is not None:

                                st.write(
                                    f"📄 {source_name} "
                                    f"— Page {page}"
                                )

                            else:

                                st.write(
                                    f"📄 {source_name}"
                                )


                # ----------------------------------
                # Save assistant message
                # ----------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )


            except Exception as e:

                error_message = (
                    "Something went wrong while "
                    "processing your question."
                )

                st.error(error_message)

                st.code(str(e))