"""
app.py
------
Kullanicinin doküman yukleyip ice aktarabildigi (ingest) ve dokumanlar
hakkinda soru sorabildigi Streamlit tabanli web arayuzu.
"""

from pathlib import Path

import streamlit as st

import ingest
from rag_engine import RAGEngine

st.set_page_config(page_title="Yerel RAG Asistani", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Yapay zeka modelleri yukleniyor (ilk calistirmada biraz surebilir)...")
def load_engine() -> RAGEngine:
    return RAGEngine()


def save_uploaded_files(uploaded_files, docs_dir: str) -> list:
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    saved_names = []
    for uploaded_file in uploaded_files:
        dest_path = Path(docs_dir) / uploaded_file.name
        with open(dest_path, "wb") as out_file:
            out_file.write(uploaded_file.getbuffer())
        saved_names.append(uploaded_file.name)
    return saved_names


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("📁 Dokuman Yonetimi")

    uploaded_files = st.file_uploader(
        "Dokuman yukleyin (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("Yuklenen Dosyalari Kaydet"):
            try:
                saved = save_uploaded_files(uploaded_files, ingest.DOCS_DIR)
                st.success(f"{len(saved)} dosya '{ingest.DOCS_DIR}/' klasorune kaydedildi.")
            except Exception as exc:
                st.error(f"Dosyalar kaydedilirken hata olustu: {exc}")

    st.divider()

    clear_existing = st.checkbox("Ice aktarmadan once veritabanini temizle", value=True)
    if st.button("🔄 Ice Aktar (Ingest)", type="primary", use_container_width=True):
        with st.spinner("Dokumanlar okunuyor, parcalaniyor ve embedding'ler hesaplaniyor (gerekirse OCR calisir)..."):
            try:
                result = ingest.ingest_documents(clear_existing=clear_existing)
                st.success(f"{result.total_chunks} parca basariyla islendi ve veritabanina kaydedildi.")
                for warning in result.warnings:
                    st.warning(warning)
                st.cache_resource.clear()
            except FileNotFoundError as exc:
                st.warning(str(exc))
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Ice aktarma sirasinda hata olustu: {exc}")

    st.divider()

    doc_count = ingest.get_chunk_count()
    st.metric("Veritabanindaki parca sayisi", doc_count)

    top_k = st.slider(
        "Sorgu basina getirilecek parca sayisi (top_k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Daha yuksek deger daha fazla baglam getirir ama yanit suresini uzatabilir.",
    )

    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("📚 Yerel RAG Asistani")
st.caption("Microsoft Foundry Local ile calisan, tamamen cevrimdisi doküman soru-cevap asistani.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Kullanilan Kaynaklar"):
                for source in message["sources"]:
                    st.write(f"- {source}")

question = st.chat_input("Dokumanlariniz hakkinda bir soru sorun...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Yanit uretiliyor..."):
            try:
                engine = load_engine()
                result = engine.answer_query(question, top_k=top_k)
            except Exception as exc:
                result = {
                    "answer": f"Beklenmeyen bir hata olustu: {exc}",
                    "sources": [],
                }
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander("📎 Kullanilan Kaynaklar"):
                    for source in result["sources"]:
                        st.write(f"- {source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )
