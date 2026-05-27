import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# ---------------- LOAD ENV ----------------

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
is_groq = "groq" in LLM_BASE_URL.lower()


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Research Paper Composer",
    page_icon="📚",
    layout="wide"
)


# ---------------- CUSTOM UI ----------------

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}

h1 {
    text-align: center;
    color: #38bdf8;
    font-size: 52px;
    font-weight: 800;
}

h2, h3 {
    color: #22c55e;
}

.stButton button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(to right, #2563eb, #06b6d4);
    color: white;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #1d4ed8, #0891b2);
}

.stTextInput input {
    border-radius: 10px;
    background-color: #1e293b;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------

st.title("📚 AI Research Paper Composer")

st.markdown("""
<div style='text-align:center;
font-size:20px;
color:#cbd5e1;
margin-bottom:20px;'>

Upload research papers and automatically generate:

<b>Abstract, Introduction, Literature Review,
Methodology, Results, Research Gaps, and Conclusion</b>

using Pure RAG Architecture.
</div>
""", unsafe_allow_html=True)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("📄 Upload Research Papers")

uploaded_files = st.sidebar.file_uploader(
    "Upload 2–6 Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("🚀 Process Papers")


# ---------------- EMBEDDING MODEL ----------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------- LLM MODEL ----------------

deprecated_groq_models = {
    "llama3-8b-8192",
    "llama3-70b-8192",
}

if is_groq:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in .env (required when LLM_BASE_URL points to Groq).")
        st.stop()

    raw_model = os.getenv("LLM_MODEL_NAME", "").strip()

    if not raw_model or raw_model in deprecated_groq_models:
        LLM_MODEL_NAME = "llama-3.3-70b-versatile"
    else:
        LLM_MODEL_NAME = raw_model

    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        timeout=120,
    )

else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error(
            "Missing OPENAI_API_KEY. Add it to the .env file as `OPENAI_API_KEY=...` and restart the app."
        )
        st.stop()

    raw_model = os.getenv("LLM_MODEL_NAME", "").strip()
    LLM_MODEL_NAME = raw_model if raw_model else "gpt-4o-mini"

    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=LLM_BASE_URL or None,
        temperature=0.3,
        timeout=120,
    )


# ---------------- MAIN PROCESS ----------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None


if process_button and uploaded_files:
    documents = []
    os.makedirs("temp", exist_ok=True)

    with st.spinner("🔄 Processing Research Papers..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            split_docs,
            embedding_model
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8}
        )

    st.success("✅ Research Papers Processed Successfully!")


if st.session_state.retriever:
    st.divider()

    def generate_response(query):
        docs = st.session_state.retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = f"""
You are an AI Research Assistant.

Use the provided research-paper context to answer accurately.

Research Context:
{context}

Question:
{query}
"""

        response = llm.invoke(final_prompt)
        return response.content

    st.header("💬 Ask Questions")

    user_question = st.text_input(
        "Ask anything from uploaded research papers"
    )

    if user_question:
        with st.spinner("Generating Answer..."):
            answer = generate_response(user_question)

        st.subheader("🤖 AI Response")
        st.write(answer)

    st.divider()

    st.header("🧠 Generated Research Paper")

    sections = {
        "📌 Abstract": "Generate a high-quality abstract using uploaded research papers.",
        "📖 Introduction": "Generate a detailed introduction using uploaded papers.",
        "📚 Literature Review": "Generate a literature review by comparing uploaded papers.",
        "⚙️ Methodology": "Extract methodologies from uploaded papers.",
        "📊 Results": "Generate a results discussion using uploaded papers.",
        "🚀 Research Gaps": "Identify future work and research gaps from uploaded papers.",
        "✅ Conclusion": "Generate a strong conclusion using uploaded papers."
    }

    for title, query in sections.items():
        with st.spinner(f"Generating {title}..."):
            response = generate_response(query)

        st.subheader(title)
        st.write(response)
        st.divider()

else:
    st.info("📄 Upload Research Papers and Click 'Process Papers'")