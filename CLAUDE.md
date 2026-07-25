# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully offline, local RAG (Retrieval-Augmented Generation) document Q&A assistant. Chat
generation runs on **Microsoft Foundry Local** (on-device LLM runtime, no cloud calls);
embedding runs on **sentence-transformers** (see "Why sentence-transformers" below — this
is a deliberate deviation from the original all-Foundry-Local design). Documents are
chunked, embedded, and stored in a local SQLite file; queries are answered by retrieving
the most similar chunks and grounding the chat model's response in them.

## Commands

There is no build step, test suite, or linter configured in this repo yet — don't assume
`pytest`, `ruff`, `make`, etc. exist unless you add them yourself.

```bash
# Setup
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Foundry Local must be installed separately and running (macOS: brew, Windows: winget)
foundry model list          # check available model aliases before running anything

# OCR fallback deps (system binaries, not pip) — macOS shown, see README for Linux/Windows
brew install tesseract poppler

# Ingest documents from docs/ into rag_store.db (CLI path)
python ingest.py                 # clears and rebuilds the DB from docs/
python ingest.py --no-clear      # append instead of clearing
python ingest.py --docs-dir X --db-path Y   # override paths

# Run the app (also lets you upload docs + trigger ingest from the UI)
streamlit run app.py

# Functional test set (answerable + unanswerable questions against RAGEngine)
cp test_questions.example.json test_questions.json   # then edit for your docs
python evaluate.py

# Quick syntax check (no pytest/ruff/make configured — evaluate.py above is the closest
# thing to a test suite)
python -m py_compile ingest.py rag_engine.py app.py evaluate.py
```

This is a git repo (initialized as part of a hardening pass; `.gitignore` excludes
`docs/*` — student's own course PDFs shouldn't be committed — plus `rag_store.db`,
`test_questions.json`, `__pycache__/`, `venv/`). There's still no CI/lint config, so don't
invent `make`/`tox`/`ruff` targets.

Model aliases and paths are overridable via env vars (see `ingest.py` top-of-file config):
`RAG_DOCS_DIR`, `RAG_DB_PATH`, `RAG_CHAT_MODEL` (default `phi-3.5-mini`, resolved against
Foundry Local's catalog), `RAG_EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`,
a Hugging Face Hub model id resolved by sentence-transformers, unrelated to Foundry Local's
catalog). If `RAG_CHAT_MODEL` doesn't match `foundry model list` output on the target machine,
Foundry Local init will fail — that's the most common runtime error for the chat half.

**`foundry-local-sdk` must stay pinned to `<1.0.0`** (see `requirements.txt`). Version 1.0.0
replaced the whole SDK with an incompatible "control-plane" design (package renamed
`foundry_local` → `foundry_local_sdk`, no more `.endpoint`/`.api_key`/`.get_model_info()`
constructor-bootstrap pattern). `rag_engine.py` targets the pre-1.0 API for the chat model.
Don't remove the upper bound to "get the latest version" — it breaks the import.

### Why sentence-transformers instead of Foundry Local for embeddings

The original design (and the tutorial this project is based on) used Foundry Local for
*both* chat and embeddings via an alias like `qwen3-embedding-0.6b`. As of this writing,
a real install of Foundry Local (v0.8.119, installed via the `microsoft/foundrylocal` brew
tap) was checked directly against its catalog (`foundry model list`, and the raw
`/foundry/list` HTTP payload) and **contained zero embedding models** — only
`chat-completion`, `vision-language-chat`, and `automatic-speech-recognition` tasks exist
in the catalog. So `ingest.py` generates embeddings with `sentence-transformers` instead;
Foundry Local is only used in `rag_engine.py` for the chat/generation step. If Foundry
Local's catalog gains embedding models again in the future, `get_embedding_model()` /
`get_embedding()` in `ingest.py` are the only functions that would need to move back to a
`FoundryLocalManager`-based implementation — nothing else in the codebase assumes where
embeddings come from.

## Architecture

Three modules form a strict dependency chain — `app.py` → `rag_engine.py` → `ingest.py`.
`rag_engine.py` deliberately imports its DB path and embedding function from `ingest.py`
rather than redefining them, so embedding logic and DB schema only exist in one place.

```
app.py (Streamlit UI)
   └─ RAGEngine (rag_engine.py)
         ├─ imports DB_PATH, get_chunk_count, get_embedding from ingest.py
         ├─ loads chat model via FoundryLocalManager(CHAT_MODEL_ALIAS)
         ├─ get_top_chunks(query): embeds query, reads all rows from `chunks` table
         │   (source, content, embedding, page_number), computes cosine similarity in
         │   Python (brute-force, numpy), returns top-k as (score, source, content, page_number)
         └─ answer_query(question): builds SYSTEM_PROMPT_TEMPLATE with retrieved
             chunks as context (citations formatted via _format_source() as
             "file.pdf (sayfa N)"), calls chat.completions.create via OpenAI-compatible
             client pointed at Foundry Local's local endpoint

ingest.py
   ├─ SentenceTransformer is a lazy singleton at module level (get_embedding_model())
   │   — reused across both ingestion and queries (rag_engine.py calls get_embedding()
   │   from here, not its own copy)
   ├─ load_documents(): reads .txt/.md/.pdf, returns {filename: [(page_number, text), ...]}
   │   (page_number is None for .txt/.md — they're treated as one unnumbered "page")
   ├─ read_pdf_file(): per-page pypdf extraction; if a page has embedded images AND
   │   extracted text is under OCR_MIN_CHARS (40 chars), falls back to _ocr_page()
   │   (pdf2image → Tesseract). Both pytesseract/pdf2image are imported lazily inside
   │   _ocr_page() so a missing Tesseract/Poppler install only disables OCR, not ingestion.
   ├─ chunk_text(): paragraph-aware chunking (~1000 chars, 150-char overlap only
   │   applied when splitting a single oversized paragraph) — applied per-page, so
   │   chunks never span a PDF page boundary (this is what makes page-numbered
   │   citations possible)
   ├─ ingest_documents() returns an `IngestResult(total_chunks, warnings)` dataclass,
   │   not a bare int — `warnings` carries per-page OCR outcomes ("OCR ile okundu" /
   │   "OCR calismadi") up to both the CLI (`main()`) and the Streamlit sidebar
   └─ SQLite table `chunks(id, source, chunk_index, content, embedding, page_number)` —
       embedding is a JSON-serialized float list, not a BLOB; `page_number` is nullable
       and `init_db()` ALTERs it onto pre-existing DBs that predate this column
```

Key behaviors that aren't obvious from skimming a single file:

- **Foundry Local integration pattern** (chat only, in `rag_engine.py`): `FoundryLocalManager(alias)`
  starts/locates the local model service; `manager.endpoint` / `manager.api_key` are then handed to
  a standard `openai.OpenAI` client, so chat calls go through the OpenAI SDK's normal
  `chat.completions.create`, not a Foundry-specific call shape. Embeddings do **not** go through
  this client — see "Why sentence-transformers" above.
- **Embeddings are pre-normalized**: `get_embedding()` in `ingest.py` calls
  `model.encode(text, normalize_embeddings=True)`, so cosine similarity in `rag_engine.py` is
  computed against unit vectors — don't add a second normalization step.
- **Anti-hallucination is enforced in the system prompt only** (`SYSTEM_PROMPT_TEMPLATE` in
  `rag_engine.py`), not via any post-processing — if you change retrieval or prompting, keep the
  "don't answer outside the provided context" instruction intact.
- **Re-ingestion is destructive by default**: `ingest_documents(clear_existing=True)` wipes the
  `chunks` table before repopulating. This matters because switching `RAG_EMBEDDING_MODEL`
  changes vector dimensionality — old and new embeddings must never coexist in the same table.
- **OCR threshold was tuned on a real failure, not guessed**: while testing this project with a
  real slide-deck PDF, one page had 6 embedded images but pypdf extracted only 12 characters of
  text (the slide title) — the actual content (logic gate diagrams/truth tables) was 100%
  raster image. `OCR_MIN_CHARS = 40` in `ingest.py` is deliberately low so it only fires on
  pages that are essentially image-only, not on normal text pages that happen to be short.
  Don't raise it casually — it'll start OCR'ing (slow) pages that already extracted fine.
- **No vector index/extension** — similarity search reads every row into memory each query. This
  is intentional for the small document sets this project targets (see README's "Sorun Giderme"
  section); don't add a vector DB dependency without discussing the tradeoff first.
- Streamlit caches the `RAGEngine` instance via `@st.cache_resource` so Foundry Local models load
  once per session; the sidebar's "Ice Aktar" button calls `st.cache_resource.clear()` after
  re-ingesting so a changed embedding model forces a fresh `RAGEngine`.

## Language note

UI strings, code comments, error messages, and README are in Turkish (project built for a
Turkish-language student program). Keep new user-facing strings consistent with that unless
told otherwise.
