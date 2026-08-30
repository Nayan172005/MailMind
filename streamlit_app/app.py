"""
MailMind — Streamlit Demo Interface

A simple, functional Streamlit UI for interacting with the RAG pipeline.

Usage:
    streamlit run streamlit_app/app.py
"""

import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from rag.pipeline import RAGPipeline


# ─── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="MailMind",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .email-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-label {
        font-size: 0.8rem;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Pipeline ─────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached across reruns)."""
    p = RAGPipeline()
    try:
        p.load_index()
    except FileNotFoundError:
        return p, False
    return p, True


pipeline, index_loaded = load_pipeline()


# ─── Session State ────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.selectbox(
        "Mode",
        ["🔍 Search", "❓ Ask", "📝 Summarize", "💬 Conversation"],
        index=1,
    )

    top_k = st.slider("Results to retrieve", 1, 20, 5)

    st.markdown("---")

    st.markdown("## 📊 Index Status")
    if index_loaded:
        status = pipeline.get_index_status()
        st.success(f"✅ Index loaded")
        st.metric("Total Vectors", status.total_vectors)
        st.metric("Total Emails", status.total_emails)
    else:
        st.warning("⚠️ No index loaded")
        if st.button("🔨 Build Index"):
            with st.spinner("Building index... This may take a few minutes."):
                try:
                    pipeline.build_index(force_rebuild=True)
                    st.success("Index built successfully! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    if mode == "💬 Conversation":
        if st.button("🗑️ Clear History"):
            pipeline.clear_conversation(st.session_state.session_id)
            st.session_state.conversation_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#666; font-size:0.8rem;'>"
        "MailMind v1.0<br>RAG-powered email assistant"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Main Content ────────────────────────────────────────────────────

st.markdown('<p class="main-header">📧 MailMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Intelligent Email Search & Question Answering</p>',
    unsafe_allow_html=True,
)

if not index_loaded:
    st.info("👆 Please build the index using the sidebar button first.")
    st.stop()


# ─── Helper Functions ────────────────────────────────────────────────

def display_results(results):
    """Display search result cards."""
    if not results:
        st.info("No relevant emails found.")
        return

    for i, r in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**{i}. {r.subject or '(No Subject)'}**")
                st.markdown(f"📤 {r.sender_name or r.sender_email} · 📅 {r.date}")
            with col2:
                score_pct = int(r.score * 100)
                st.markdown(
                    f'<span class="score-badge">{score_pct}%</span>',
                    unsafe_allow_html=True,
                )

            if r.snippet:
                with st.expander("Preview"):
                    st.text(r.snippet[:500])

            st.markdown("---")


# ─── Query Input ──────────────────────────────────────────────────────

if mode == "💬 Conversation":
    # Show conversation history
    for msg in st.session_state.conversation_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    query = st.chat_input("Ask a follow-up question...")

    if query:
        st.chat_message("user").write(query)

        with st.spinner("Thinking..."):
            try:
                response = pipeline.conversation_ask(
                    session_id=st.session_state.session_id,
                    query=query,
                    top_k=top_k,
                )

                st.session_state.conversation_history.append(
                    {"role": "user", "content": query}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response.answer}
                )

                st.chat_message("assistant").write(response.answer)

                if response.sources:
                    with st.expander(f"📎 Sources ({len(response.sources)} emails)"):
                        display_results(response.sources)

            except Exception as e:
                st.error(f"Error: {e}")

else:
    query = st.text_input(
        "Enter your query:",
        placeholder=(
            "e.g., 'Find emails about placement interviews' or "
            "'When is my Microsoft assessment?'"
        ),
    )

    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        submit = st.button("🚀 Go", use_container_width=True)

    if submit and query:
        with st.spinner("Processing..."):
            try:
                if mode == "🔍 Search":
                    response = pipeline.search(query, top_k=top_k)
                    st.markdown(f"### Results ({response.total})")
                    display_results(response.results)

                elif mode == "❓ Ask":
                    response = pipeline.ask(query, top_k=top_k)

                    st.markdown("### Answer")
                    st.markdown(response.answer)

                    if response.sources:
                        st.markdown("---")
                        with st.expander(f"📎 Sources ({len(response.sources)} emails)"):
                            display_results(response.sources)

                elif mode == "📝 Summarize":
                    response = pipeline.summarize(query=query, top_k=top_k)

                    st.markdown("### Summary")
                    st.markdown(response.summary)

                    if response.sources:
                        st.markdown("---")
                        with st.expander(f"📎 Sources ({len(response.sources)} emails)"):
                            display_results(response.sources)

            except Exception as e:
                st.error(f"Error: {e}")

    elif submit:
        st.warning("Please enter a query.")
