"""
ingest.py
---------
Dokumanlari (.txt, .md, .pdf) okur, parcalara (chunks) boler, sentence-transformers
uzerinden embedding vektorlerini hesaplar ve SQLite veritabanina kaydeder.

Not: Embedding icin Foundry Local degil, yerel olarak calisan sentence-transformers
kullanilir. Bu proje yazilirken kontrol edildiginde Foundry Local'in model katalogunda
(`foundry model list`) hicbir embedding modeli bulunmuyordu (yalnizca chat-completion,
vision-language-chat ve automatic-speech-recognition gorevleri mevcuttu). sentence-transformers
modeli ilk calistirmada bir kez internetten indirilir, sonrasinda tamamen cevrimdisi calisir.
Foundry Local, sohbet/uretim adiminda (bkz. rag_engine.py) kullanilmaya devam eder.

Bu modul rag_engine.py tarafindan da (DB baglantisi ve embedding fonksiyonlari
icin) import edilir.
"""

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Yapilandirma
# --------------------------------------------------------------------------
DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "docs")
DB_PATH = os.environ.get("RAG_DB_PATH", "rag_store.db")

# Foundry Local sohbet modeli alias'i (rag_engine.py tarafindan kullanilir).
# Kendi makinenizde `foundry model list` komutuyla mevcut alias'lari kontrol edip
# gerekirse degistirin.
CHAT_MODEL_ALIAS = os.environ.get("RAG_CHAT_MODEL", "phi-3.5-mini")

# Embedding icin kullanilan sentence-transformers modeli (Hugging Face Hub adi).
EMBEDDING_MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# PDF sayfalarinda gomulu resim var ve metin katmani bu esigin altindaysa (karakter
# cinsinden) OCR denenir. Kucuk deger: sadece "neredeyse hic metin yok" durumunu yakalar,
# normal (metin agirlikli) sayfalari gereksiz yere OCR'a sokmaz.
OCR_MIN_CHARS = 40
OCR_LANGUAGE = os.environ.get("RAG_OCR_LANG", "eng")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class IngestResult:
    """ingest_documents() sonucu: islenen parca sayisi + kalite uyarilari (ör. basarisiz/zayif OCR)."""

    total_chunks: int
    warnings: list = field(default_factory=list)

# --------------------------------------------------------------------------
# Embedding modeli (lazy singleton)
# --------------------------------------------------------------------------
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """sentence-transformers embedding modelini yukler (ilk cagrida bir kez)."""
    global _embedding_model

    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding modeli yuklenemedi ('{EMBEDDING_MODEL_NAME}'). Model ilk "
                f"kullanimda internetten indirilir; internet baglantinizi kontrol edin "
                f"veya model adini dogrulayin. Detay: {exc}"
            ) from exc

    return _embedding_model


def get_embedding(text: str) -> list:
    """Verilen metin icin embedding vektoru hesaplar."""
    if not text or not text.strip():
        raise ValueError("Embedding hesaplanacak metin bos olamaz.")

    model = get_embedding_model()
    try:
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    except Exception as exc:
        raise RuntimeError(f"Embedding olusturulurken hata olustu: {exc}") from exc


# --------------------------------------------------------------------------
# Veritabani islemleri
# --------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """SQLite veritabanini ve gerekli tabloyu olusturur, baglantiyi dondurur.

    Eski (page_number kolonu olmayan) bir rag_store.db dosyasiyla karsilasilirsa
    kolonu ekler, boylece kullanicinin mevcut veritabanini elle silmesi gerekmez.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            page_number INTEGER
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "page_number" not in existing_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER")
    conn.commit()
    return conn


def clear_chunks(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    source: str,
    chunk_index: int,
    content: str,
    embedding: list,
    page_number: int = None,
) -> None:
    conn.execute(
        "INSERT INTO chunks (source, chunk_index, content, embedding, page_number) VALUES (?, ?, ?, ?, ?)",
        (source, chunk_index, content, json.dumps(embedding), page_number),
    )


def get_chunk_count(db_path: str = DB_PATH) -> int:
    """Veritabanindaki toplam parca sayisini dondurur. DB yoksa 0 doner."""
    if not Path(db_path).exists():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Dokuman okuma
# --------------------------------------------------------------------------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"'{path.name}' okunurken hata olustu: {exc}") from exc


def _page_needs_ocr(text: str, page) -> bool:
    """Sayfada gomulu resim var VE metin katmani neredeyse bossa OCR adayidir.

    Gercek bir vakada test edildi: 6 gomulu resim iceren bir slayt sadece 12 karakter
    metin veriyordu (sadece baslik), asil icerik (diyagram/tablo) tamamen resimdi.
    """
    try:
        has_images = bool(page.images)
    except Exception:
        has_images = False
    return has_images and len(text.strip()) < OCR_MIN_CHARS


def _ocr_page(path: Path, page_number: int) -> str:
    """Tek bir PDF sayfasini resme cevirip Tesseract ile okur.

    pytesseract/pdf2image importu bilerek burada (lazy) yapiliyor: paket veya
    tesseract/poppler binary'si kurulu degilse sadece OCR devre disi kalsin,
    ingest.py'nin geri kalani (plain-text extraction) calismaya devam etsin.
    """
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(path), first_page=page_number, last_page=page_number, dpi=200)
    return "\n".join(pytesseract.image_to_string(img, lang=OCR_LANGUAGE) for img in images)


def read_pdf_file(path: Path, warnings: list) -> list:
    """PDF'i sayfa sayfa okur. [(sayfa_no, metin), ...] dondurur (1-indeksli sayfa no).

    Metin katmani cok zayif ve sayfada gomulu resim varsa OCR denenir; OCR daha uzun/
    anlamli metin verirse o kullanilir. Hem plain-text hem OCR basarisiz olsa da akis
    durmaz, sadece warnings listesine bilgi eklenir.
    """
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"'{path.name}' PDF olarak acilamadi: {exc}") from exc

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Bozuk tek bir sayfa yuzunden tum dokumani atlamayalim.
            text = ""

        if _page_needs_ocr(text, page):
            try:
                ocr_text = _ocr_page(path, page_number)
            except Exception as exc:
                ocr_text = ""
                warnings.append(
                    f"'{path.name}' sayfa {page_number}: metin katmani cok zayif "
                    f"(muhtemelen gorsel/diyagram) ve OCR calismadi ({exc}). Bu sayfanin "
                    f"icerigi eksik kalabilir."
                )
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
                warnings.append(
                    f"'{path.name}' sayfa {page_number}: metin katmani zayifti, "
                    f"OCR ile okundu."
                )

        pages.append((page_number, text))

    return pages


def load_documents(docs_dir: Path, warnings: list) -> dict:
    """docs_dir icindeki desteklenen dosyalari okuyup {dosya_adi: [(sayfa_no, metin), ...]} dondurur.

    .txt/.md dosyalari icin sayfa_no None'dur (tek parca metin olarak islenir).
    """
    documents = {}
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            pages = read_pdf_file(path, warnings)
        else:
            text = read_text_file(path)
            pages = [(None, text)]

        pages = [(page_number, text) for page_number, text in pages if text and text.strip()]
        if pages:
            documents[path.name] = pages
    return documents


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Metni paragraf sinirlarina saygi gostererek ~chunk_size boyutunda parcalara boler."""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end])
                if end >= len(paragraph):
                    break
                start = end - overlap
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


# --------------------------------------------------------------------------
# Ana ingestion pipeline'i
# --------------------------------------------------------------------------
def ingest_documents(docs_dir: str = DOCS_DIR, db_path: str = DB_PATH, clear_existing: bool = True) -> IngestResult:
    """
    docs_dir icindeki dokumanlari okur, sayfa sinirlari icinde parcalar, embedding
    cikarir ve db_path veritabanina kaydeder. Toplam parca sayisi + kalite
    uyarilarini (ör. basarisiz OCR) iceren bir IngestResult dondurur.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists() or not any(docs_path.iterdir()):
        raise FileNotFoundError(
            f"'{docs_dir}' klasoru bulunamadi veya bos. Lutfen once dokuman ekleyin."
        )

    warnings = []
    documents = load_documents(docs_path, warnings)
    if not documents:
        raise ValueError(
            f"'{docs_dir}' klasorunde desteklenen formatta (.txt, .md, .pdf) dokuman bulunamadi."
        )

    conn = init_db(db_path)
    total_chunks = 0
    try:
        if clear_existing:
            clear_chunks(conn)

        for filename, pages in documents.items():
            chunk_idx = 0
            for page_number, page_text in pages:
                for chunk in chunk_text(page_text):
                    embedding = get_embedding(chunk)
                    insert_chunk(conn, filename, chunk_idx, chunk, embedding, page_number)
                    chunk_idx += 1
                    total_chunks += 1
            conn.commit()
    finally:
        conn.close()

    return IngestResult(total_chunks=total_chunks, warnings=warnings)


# --------------------------------------------------------------------------
# CLI giris noktasi
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Dokumanlari SQLite veritabanina ingest eder.")
    parser.add_argument("--docs-dir", default=DOCS_DIR, help="Dokumanlarin bulundugu klasor.")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite veritabani dosya yolu.")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Mevcut verileri silmeden yeni parcalari ekler (varsayilan: mevcut veriler silinir).",
    )
    args = parser.parse_args()

    try:
        result = ingest_documents(args.docs_dir, args.db_path, clear_existing=not args.no_clear)
        for warning in result.warnings:
            print(f"[UYARI] {warning}")
        print(f"[OK] {result.total_chunks} parca basariyla islendi ve '{args.db_path}' veritabanina kaydedildi.")
    except Exception as exc:
        print(f"[HATA] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
