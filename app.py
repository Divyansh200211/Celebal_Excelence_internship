from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import RAGChatbot
from src.config import AppConfig
from src.logger import get_logger
from src.metrics import write_metrics_report
from src.vectorstore import DocumentStore

load_dotenv()
logger = get_logger("app")

st.set_page_config(page_title="Document QA with RAG", page_icon="📄", layout="wide")


def build_chatbot(config: AppConfig | None = None) -> RAGChatbot:
    return RAGChatbot(config or AppConfig())


def build_store(config: AppConfig | None = None) -> DocumentStore:
    return DocumentStore(config or AppConfig())


def save_uploaded_files(uploaded_files) -> List[Path]:
    saved_paths: List[Path] = []
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as handle:
            handle.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def reset_session_state() -> None:
    st.session_state.messages = []
    st.session_state.knowledge_ready = False


def main() -> None:
    st.title("📄 Document Question Answering with RAG")
    st.caption("Upload PDFs or TXT files, build a vector knowledge base, and ask grounded questions.")
    st.info("This session is temporary. Your uploaded documents and chat history are cleared when the app is closed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_ready" not in st.session_state:
        st.session_state.knowledge_ready = False

    st.subheader("API Credentials")
    cohere_key = st.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY", ""))
    pinecone_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))

    config = AppConfig()
    sanitized_cohere_key = cohere_key.strip().strip('"').strip("'") if cohere_key else ""
    sanitized_pinecone_key = pinecone_key.strip().strip('"').strip("'") if pinecone_key else ""
    if sanitized_cohere_key:
        os.environ["COHERE_API_KEY"] = sanitized_cohere_key
        config.cohere_api_key = sanitized_cohere_key
    if sanitized_pinecone_key:
        os.environ["PINECONE_API_KEY"] = sanitized_pinecone_key
        config.pinecone_api_key = sanitized_pinecone_key

    if not config.cohere_api_key or not config.pinecone_api_key:
        st.warning("Enter your Cohere and Pinecone API keys above to continue.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload one or more PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    build_col, clear_col = st.columns([1, 1])
    with build_col:
        if st.button("Build Knowledge Base", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF or TXT file.")
                st.stop()

            with st.spinner("Processing documents and building the vector store..."):
                saved_paths = save_uploaded_files(uploaded_files)
                store = build_store(config)
                documents, total_chars, total_pages = store.load_documents(saved_paths)
                if not documents:
                    st.error("No valid content could be extracted from the uploaded files.")
                    st.stop()
                store.build_index(documents)

                metrics = {
                    "document_count": len(saved_paths),
                    "page_count": total_pages,
                    "character_count": total_chars,
                    "chunk_size": AppConfig().chunk_size,
                    "chunk_overlap": AppConfig().chunk_overlap,
                    "total_chunks": len(documents),
                    "embedding_model": AppConfig().embedding_model,
                    "embedding_dimensions": store.embedding_dim,
                    "vector_db": AppConfig().vector_db_path,
                    "llm_model": AppConfig().llm_model,
                    "top_k": AppConfig().top_k,
                    "avg_retrieval_time": 0.0,
                    "avg_generation_time": 0.0,
                }
                write_metrics_report(metrics)
                st.session_state.knowledge_ready = True
                st.success("Knowledge base built successfully.")

    with clear_col:
        if st.button("Clear Chat", use_container_width=True):
            reset_session_state()
            st.success("Chat history cleared.")

    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded documents")
    if question:
        if not st.session_state.knowledge_ready:
            st.warning("Build the knowledge base first before asking questions.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            store = build_store(config)
            chatbot = build_chatbot(config)
            results = store.search(question, top_k=AppConfig().top_k)
            response = chatbot.answer_question(question, results)

            assistant_reply = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
                st.caption(f"Response time: {response['response_time']:.2f}s")
        except Exception as exc:
            logger.exception("Question handling failed: %s", exc)
            error_message = str(exc)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)


if __name__ == "__main__":
    main()
