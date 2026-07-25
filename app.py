"""
app.py
------
Kullanicinin doküman yukleyip ice aktarabildigi (ingest), yuklenen dokumanlari
tek tek yonetip silebildigi ve dokumanlar hakkinda soru sorabildigi Streamlit
tabanli web arayuzu.
"""

import html
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

# Temel koyu tema (arka plan/metin/renk kontrasti) .streamlit/config.toml uzerinden
# Streamlit'in kendi tema motoruyla ayarlanir (tum widget'larda tutarli sonuc verir).
# Buradaki CSS ise "Fluent / modern SaaS panel" hissi veren butik detaylari
# (kart, pill rozet, hover animasyonu, gradient buton) ekler.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --rag-bg-card: #1e293b;
    --rag-border: #334155;
    --rag-text: #f1f5f9;
    --rag-text-muted: #94a3b8;
    --rag-accent: #2563eb;
    --rag-accent-light: #60a5fa;
    --rag-accent-gradient: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    --rag-radius: 12px;
    --rag-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* Streamlit'in varsayilan hamburger menusu / footer'ini gizleyerek daha 'urun' hissi ver */
#MainMenu, footer {
    visibility: hidden;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--rag-border);
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* --- Metin/stat kart (parca sayisi) --- */
[data-testid="stMetric"] {
    background: var(--rag-bg-card);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius);
    padding: 16px 18px;
    box-shadow: var(--rag-shadow);
}
[data-testid="stMetricValue"] {
    color: var(--rag-accent-light);
}
[data-testid="stMetricLabel"] {
    color: var(--rag-text-muted);
}

/* --- Butonlar: varsayilan olarak hafif kart gibi, hover'da yukari kayar --- */
.stButton > button {
    border-radius: var(--rag-radius);
    border: 1px solid var(--rag-border);
    background-color: var(--rag-bg-card);
    color: var(--rag-text);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-3px);
    border-color: var(--rag-accent-light);
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
    color: var(--rag-text);
}
.stButton > button:active {
    transform: translateY(-1px);
}

/* --- Birincil buton (Ice Aktar): gradient accent, daha belirgin --- */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--rag-accent-gradient);
    border: none;
    color: #ffffff;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.45);
}

/* --- Sohbet input ve mesaj balonlari --- */
[data-testid="stChatInput"] textarea {
    border-radius: var(--rag-radius);
}
[data-testid="stChatMessage"] {
    background: var(--rag-bg-card);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius);
    box-shadow: var(--rag-shadow);
    margin-bottom: 10px;
}

/* --- Genisletilebilir bolumler (kaynaklar) --- */
[data-testid="stExpander"] {
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius);
    overflow: hidden;
}

/* --- Dosya yukleyici --- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: var(--rag-radius);
}

/* --- Bildirim kutulari (durum rozeti, uyarilar) --- */
[data-testid="stAlertContainer"] {
    border-radius: var(--rag-radius);
}

/* --- Kaynak karti + pill rozetler (render_citations icinde uretilir) --- */
.rag-source-card {
    background: var(--rag-bg-card);
    border: 1px solid var(--rag-border);
    border-radius: var(--rag-radius);
    padding: 14px 16px;
    margin-bottom: 10px;
}
.rag-source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}
.rag-source-name {
    font-weight: 600;
    color: var(--rag-text);
}
.rag-pill-group {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.rag-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
}
.rag-pill-page {
    background: rgba(37, 99, 235, 0.15);
    color: var(--rag-accent-light);
    border: 1px solid rgba(37, 99, 235, 0.35);
}
.rag-pill-score {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.rag-source-snippet {
    color: var(--rag-text-muted);
    font-size: 0.9rem;
    line-height: 1.55;
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    kesiti) sekilli kart + pill rozetlerle gosteren genisletilebilir bir bolum ekler.

    Kaynak/metin icerigi kullanicinin kendi dokumanlarindan (OCR dahil) geldigi
    icin ham HTML'e gomulmeden once html.escape() ile kacislanir - aksi halde
    metin icinde gecen '<'/'>' gibi karakterler duzeni bozabilirdi.
    """
    if not retrieved_chunks:
        return
    with st.expander("📚 Yararlanilan Kaynaklar ve Metin Parcalari"):
        for score, source, content, page_number in retrieved_chunks:
            safe_source = html.escape(source)
            snippet = content if len(content) <= 400 else content[:400] + "…"
            safe_snippet = html.escape(snippet.replace("\n", " "))

            page_pill = (
                f'<span class="rag-pill rag-pill-page">📄 Sayfa {page_number}</span>'
                if page_number
                else ""
            )
            score_pill = f'<span class="rag-pill rag-pill-score">🎯 {score:.2f} benzerlik</span>'

            st.markdown(
                f"""
                <div class="rag-source-card">
                    <div class="rag-source-header">
                        <span class="rag-source-name">{safe_source}</span>
                        <div class="rag-pill-group">{page_pill}{score_pill}</div>
                    </div>
                    <div class="rag-source-snippet">{safe_snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
