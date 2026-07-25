"""
app.py
------
Kullanicinin doküman yukleyip ice aktarabildigi (ingest), yuklenen dokumanlari
tek tek yonetip silebildigi ve dokumanlar hakkinda soru sorabildigi Streamlit
tabanli web arayuzu.
"""

from pathlib import Path

import streamlit as st

import ingest
from rag_engine import RAGEngine

st.set_page_config(page_title="Yerel RAG Asistani", page_icon="📚", layout="wide")

EXAMPLE_QUESTIONS = [
    "Yuklenen belgelerin ana konulari nelerdir?",
    "Dokumanlardaki onemli tarih ve tanimlari ozetle.",
    "Bu belgedeki en onemli 3 noktayi maddele.",
    "Bu dokumanla ilgili sikca sorulabilecek bir soru oner ve cevapla.",
]


@st.cache_resource(show_spinner="Yapay zeka modelleri yukleniyor (ilk calistirmada biraz surebilir)...")
def load_engine() -> RAGEngine:
    return RAGEngine()


def get_engine() -> RAGEngine:
    """load_engine()'i cagirir ve sonucu sidebar'daki durum rozetine yansitir."""
    try:
        engine = load_engine()
        st.session_state.engine_status = "ready"
        return engine
    except Exception:
        st.session_state.engine_status = "error"
        raise


def save_uploaded_files(uploaded_files, docs_dir: str) -> list:
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    saved_names = []
    for uploaded_file in uploaded_files:
        dest_path = Path(docs_dir) / uploaded_file.name
        with open(dest_path, "wb") as out_file:
            out_file.write(uploaded_file.getbuffer())
        saved_names.append(uploaded_file.name)
    return saved_names


def render_citations(retrieved_chunks: list) -> None:
    """Bir yanitin altina, kullanilan kaynaklari (dosya, sayfa, benzerlik, metin
    kesiti) gosteren genisletilebilir bir bolum ekler."""
    if not retrieved_chunks:
        return
    with st.expander("📚 Yararlanilan Kaynaklar ve Metin Parcalari"):
        for score, source, content, page_number in retrieved_chunks:
            label = f"{source} (sayfa {page_number})" if page_number else source
            st.markdown(f"**{label}**  \n*Benzerlik skoru: {score:.2f}*")
            snippet = content if len(content) <= 400 else content[:400] + "…"
            st.markdown(f"> {snippet.replace(chr(10), ' ')}")
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine_status" not in st.session_state:
    st.session_state.engine_status = "not_loaded"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    # --- Sistem durumu rozeti ---
    engine_status = st.session_state.engine_status
    if engine_status == "ready":
        st.success(f"🟢 Foundry Local LLM Aktif | Model: {ingest.CHAT_MODEL_ALIAS}")
    elif engine_status == "error":
        st.error(f"🔴 Foundry Local baslatilamadi | Model: {ingest.CHAT_MODEL_ALIAS}")
    else:
        st.info(f"🟡 Foundry Local hazir | Model: {ingest.CHAT_MODEL_ALIAS} (ilk soruda yuklenecek)")

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

    st.caption("📚 Yuklenen Dokumanlar")
    document_sources = ingest.get_document_sources()
    if document_sources:
        for source in document_sources:
            doc_col, delete_col = st.columns([5, 1])
            doc_col.write(f"📄 {source}")
            if delete_col.button("🗑️", key=f"delete_{source}", help=f"'{source}' dosyasini veritabanindan sil"):
                try:
                    deleted_count = ingest.delete_document(source)
                    st.success(f"'{source}' veritabanindan silindi ({deleted_count} parca).")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        st.caption("Not: Silme islemi yalnizca veritabanindan kaldirir; dosya 'docs/' klasorunde kalir.")
    else:
        st.caption("Henuz veritabaninda doküman yok.")

    st.divider()

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

pending_question = st.session_state.pop("pending_question", None)

if not st.session_state.messages and not pending_question:
    st.markdown("**Hizli baslangic icin ornek bir soru secin:**")
    example_cols = st.columns(2)
    for i, example_question in enumerate(EXAMPLE_QUESTIONS):
        if example_cols[i % 2].button(example_question, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = example_question
            st.rerun()
    st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_citations(message.get("retrieved_chunks", []))

chat_input_question = st.chat_input("Dokumanlariniz hakkinda bir soru sorun...")
question = pending_question or chat_input_question

if question:
    st.session_state.messages.append({"role": "user", "content": question, "retrieved_chunks": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Yanit uretiliyor..."):
            try:
                engine = get_engine()
                result = engine.answer_query(question, top_k=top_k)
            except Exception as exc:
                result = {
                    "answer": f"Beklenmeyen bir hata olustu: {exc}",
                    "retrieved_chunks": [],
                }
            st.markdown(result["answer"])
            render_citations(result.get("retrieved_chunks", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }
    )
    # Sidebar'daki durum rozeti bu turda okunan eski engine_status ile render edildi;
    # yeniden calistirip guncel (ör. "ready") degeri hemen yansitiyoruz.
    st.rerun()
