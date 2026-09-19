import streamlit as st
from pathlib import Path
import rag_engine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AcademiaRAG Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "university_docs"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-top: 0px;
            margin-bottom: 25px;
        }

        .answer-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            background-color: #f8f9fa;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        .source-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: #ffffff;
            margin-top: 10px;
        }

        .verified {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AcademiaRAG Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise Knowledge Suite for University Regulations</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.markdown(
        """
        **AcademiaRAG Pro** uses:

        - 📄 Official university documents
        - 🧩 Semantic chunking
        - 🧠 Sentence Transformers
        - 🔎 FAISS vector search
        - 🎯 CrossEncoder reranking
        - ✅ Evidence verification
        - 🧠 Feedback-based learning
        """
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if pdf_files:
        pdf_names = [pdf.name for pdf in pdf_files]

        selected_pdf_name = st.selectbox(
            "Select document",
            pdf_names
        )

        selected_pdf = PDF_DIR / selected_pdf_name

    else:
        selected_pdf = None
        st.error("No PDF documents found in university_docs/")

    st.divider()

    st.subheader("ℹ️ About")

    st.caption(
        "AcademiaRAG Pro retrieves information from the indexed university "
        "knowledge base and provides evidence-backed answers."
    )


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_path):

    # Load embedding model and reranker
    rag_engine.load_models()

    # Initialize SQLite learning database
    rag_engine.init_database()

    # Build vector database if it does not already exist
    if getattr(rag_engine, "_index", None) is None:

        rag_engine.build_knowledge_base(
            str(pdf_path)
        )

    return True


# ============================================================
# START SYSTEM
# ============================================================

if selected_pdf is None:

    st.warning(
        "Please place an official university PDF inside the "
        "`university_docs` folder."
    )

    st.stop()


with st.spinner(
    "🔄 Initializing AcademiaRAG Pro... This may take a moment on first launch."
):

    try:

        initialize_rag(
            str(selected_pdf)
        )

        system_ready = True

    except Exception as e:

        system_ready = False

        st.error(
            "Failed to initialize the RAG system."
        )

        st.exception(e)


# ============================================================
# SYSTEM STATUS
# ============================================================

if system_ready:

    col1, col2, col3, col4 = st.columns(4)

    # Number of indexed chunks
    index = getattr(rag_engine, "_index", None)

    if index is not None:
        chunk_count = index.ntotal
    else:
        chunk_count = 0

    # Learning memory
    try:
        learning_count = rag_engine.get_learning_count()
    except Exception:
        learning_count = 0

    col1.metric(
        "System",
        "Ready"
    )

    col2.metric(
        "Indexed Chunks",
        chunk_count
    )

    col3.metric(
        "Learning Memory",
        learning_count
    )

    col4.metric(
        "Knowledge Source",
        "Vignan"
    )


# ============================================================
# MAIN QUESTION AREA
# ============================================================

st.divider()

st.subheader("🔎 Ask AcademiaRAG Pro")

st.write(
    "Ask questions about the indexed university regulations "
    "using natural language."
)

question = st.text_input(
    "Enter your question",
    placeholder="Example: What percentage of attendance is required in each course?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🚀 Ask AcademiaRAG Pro",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🔍 Searching the university knowledge base..."
        ):

            try:

                result = rag_engine.semantic_search(
                    question.strip()
                )

                st.session_state["last_question"] = question.strip()
                st.session_state["last_result"] = result

            except Exception as e:

                st.error(
                    "An error occurred while processing your question."
                )

                st.exception(e)


# ============================================================
# DISPLAY ANSWER
# ============================================================

if "last_result" in st.session_state:

    result = st.session_state["last_result"]

    answer = result.get(
        "answer",
        "NO RELEVANT INFORMATION FOUND"
    )

    source = result.get(
        "source",
        "Unknown"
    )

    page = result.get(
        "page",
        "Unknown"
    )

    verified = result.get(
        "verified",
        False
    )

    score = result.get(
        "score",
        0
    )

    intent = result.get(
        "intent",
        "general"
    )

    learned = result.get(
        "learned",
        False
    )

    learning_similarity = result.get(
        "learning_similarity",
        None
    )


    st.divider()

    st.subheader("💡 Answer")

    st.markdown(
        f"""
        <div class="answer-box">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # EVIDENCE / SOURCE INFORMATION
    # ========================================================

    st.subheader("📚 Evidence & Source")

    source_col1, source_col2 = st.columns(2)

    with source_col1:

        st.markdown(
            f"""
            <div class="source-box">

            **📄 Source**

            {source}

            **📖 Page**

            {page}

            </div>
            """,
            unsafe_allow_html=True
        )


    with source_col2:

        if verified:

            verification_text = "✅ Verified"

        else:

            verification_text = "⚠️ Not Verified"


        st.markdown(
            f"""
            <div class="source-box">

            **Evidence Status**

            {verification_text}

            **Retrieval Score**

            {score:.3f}

            **Detected Intent**

            {intent}

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # LEARNING INFORMATION
    # ========================================================

    if learned:

        st.info(
            f"🧠 This response was influenced by a previously learned "
            f"feedback interaction."
        )

        if learning_similarity is not None:

            st.caption(
                f"Learning similarity: {learning_similarity:.3f}"
            )


    # ========================================================
    # FEEDBACK
    # ========================================================

    st.subheader("🧠 Was this answer helpful?")

    feedback_col1, feedback_col2 = st.columns(2)


    with feedback_col1:

        if st.button(
            "👍 Helpful",
            use_container_width=True,
            key="helpful_button"
        ):

            try:

                rag_engine.save_feedback(
                    st.session_state["last_question"],
                    answer,
                    source,
                    page,
                    1
                )

                st.success(
                    "Thank you! Your feedback was saved."
                )

            except Exception as e:

                st.error(
                    "Could not save feedback."
                )

                st.exception(e)


    with feedback_col2:

        if st.button(
            "👎 Not Helpful",
            use_container_width=True,
            key="not_helpful_button"
        ):

            try:

                rag_engine.save_feedback(
                    st.session_state["last_question"],
                    answer,
                    source,
                    page,
                    0
                )

                st.warning(
                    "Feedback recorded. Thank you."
                )

            except Exception as e:

                st.error(
                    "Could not save feedback."
                )

                st.exception(e)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader("💬 Example Questions")

example_questions = [
    "What percentage of attendance is required in each course?",
    "What is the maximum attendance shortage that can be condoned?",
    "Can medical reasons be considered for attendance shortage?",
    "How often is attendance reviewed?",
    "What does an R grade mean?",
    "What are supplementary examinations?"
]

for example in example_questions:

    st.markdown(
        f"• `{example}`"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AcademiaRAG Pro • Local RAG Backend • FAISS • "
    "Sentence Transformers • CrossEncoder • SQLite"
)