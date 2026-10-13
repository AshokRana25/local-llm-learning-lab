"""Run with: python -m streamlit run app.py"""

from pathlib import Path

import streamlit as st

from src.ollama_client import DEFAULT_MODEL, OllamaError, list_models, stream_chat
from src.rag import MAX_QUESTION_CHARS, RAGStore, build_messages


@st.cache_resource
def get_store() -> RAGStore:
    return RAGStore()


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("View retrieved excerpts"):
        st.caption("These are the excerpts sent to the model. Distance is a ranking measure, not confidence.")
        for hit in sources:
            meta = hit["metadata"]
            page = f"page {meta['page']}" if meta["source"].lower().endswith(".pdf") else "TXT"
            st.text(f"[{hit['citation']}] {meta['source']} | {page} | chunk {meta['chunk']}")
            st.caption(f"Cosine distance: {hit['distance']:.3f} (smaller is closer)")
            st.text(hit["text"])


def main() -> None:
    st.set_page_config(page_title="Local LLM Learning Lab", page_icon="📚", layout="wide")
    st.title("Local LLM Learning Lab")
    st.caption("Chat locally. Add your documents. Inspect the evidence behind an answer.")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("models", [DEFAULT_MODEL])

    try:
        store = get_store()
    except Exception as exc:
        st.error(f"Could not open the local document library: {exc}")
        st.info("See README.md → Troubleshooting. Close other copies of this app before retrying.")
        st.stop()

    with st.sidebar:
        st.header("1 · Choose your model")
        if st.button("Check Ollama / refresh models", width="stretch"):
            try:
                available = list_models()
                # Exclude explicitly named cloud models from this local learning UI.
                local = [name for name in available if "cloud" not in name.lower()]
                st.session_state.models = list(dict.fromkeys([DEFAULT_MODEL] + local))
                if DEFAULT_MODEL in local:
                    st.success("Ollama is ready. The default model is installed.")
                else:
                    st.warning("Download the default model: ollama pull llama3.2:3b")
            except OllamaError as exc:
                st.error(str(exc))
        model = st.selectbox("Model", st.session_state.models)
        mode = st.radio("Mode", ["Chat", "Ask my documents (RAG)"])
        top_k = st.slider("Retrieved chunks", 1, 5, 3,
                          help="Used in document mode. More chunks can add noise.")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
        st.caption("4,096-token context · up to 512 output tokens · embeddings on CPU")

        st.header("2 · Build your library")
        uploaded = st.file_uploader("PDF or UTF-8 TXT files", type=["pdf", "txt"],
                                    accept_multiple_files=True)
        if st.button("Index uploaded files", disabled=not uploaded, width="stretch"):
            for file in uploaded or []:
                try:
                    with st.spinner(f"Indexing {file.name}… First use downloads the embedding model."):
                        result = store.add_document(file.name, file.getvalue())
                    action = "Already indexed" if result["skipped"] else "Indexed"
                    st.success(f"{action}: {result['name']} ({result['chunks']} chunks)")
                    for warning in result["warnings"]:
                        st.warning(warning)
                except Exception as exc:
                    st.error(f"Could not index {file.name}: {exc}")
        if st.button("Load the sample document", width="stretch"):
            try:
                sample = Path(__file__).parent / "documents" / "example.txt"
                with st.spinner("Indexing the sample. First use downloads the embedding model…"):
                    result = store.add_document(sample.name, sample.read_bytes())
                st.success("Sample is ready. Switch to document mode and ask about the book club.")
            except Exception as exc:
                st.error(f"Could not load the sample: {exc}")

        docs = store.documents()
        st.caption(f"{len(docs)} documents · {store.count()} stored chunks")
        if docs:
            with st.expander("Manage indexed documents"):
                choices = {doc["id"]: f"{doc['name']} · {doc['id'][:8]} · {doc['chunks']} chunks" for doc in docs}
                chosen = st.selectbox("Document version", list(choices), format_func=choices.get)
                if st.button("Remove selected document"):
                    store.remove_document(chosen)
                    st.rerun()
        st.caption("The library persists in data/chroma. Chat history lasts for this browser session.")
        if st.button("Clear chat history", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if mode == "Ask my documents (RAG)":
        st.info("Ask a complete question each time: document retrieval does not use earlier chat turns. "
                "Try: On what day and at what time does the Lantern Book Club meet?")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    prompt = st.chat_input("Ask a question…", max_chars=MAX_QUESTION_CHARS)
    if not prompt:
        return
    if mode == "Ask my documents (RAG)" and not store.count():
        st.warning("Index a document or load the sample before asking document questions.")
        return
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding relevant excerpts…" if mode != "Chat" else "Connecting to Ollama…"):
                hits = store.retrieve(prompt, top_k) if mode != "Chat" else None
                # Only plain-chat exchanges become plain-chat memory.
                history = [m for m in st.session_state.messages if m.get("mode") == "Chat"]
                messages, sources = build_messages(prompt, history, hits)
            answer = st.write_stream(stream_chat(messages, model=model, temperature=temperature))
            if not isinstance(answer, str) or not answer.strip():
                raise OllamaError("The model returned no text. Please retry.")
            show_sources(sources)
        except Exception as exc:
            st.error(f"Answer failed: {exc}")
            st.caption("Any partial text above is incomplete and was not saved. Resend your question to retry.")
            return
    st.session_state.messages.extend([
        {"role": "user", "content": prompt, "mode": mode},
        {"role": "assistant", "content": answer, "mode": mode, "sources": sources},
    ])


if __name__ == "__main__":
    main()
