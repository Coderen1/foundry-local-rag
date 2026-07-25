"""
rag_engine.py
-------------
Kullanicinin sorusunu alir, embedding'ini cikarir, SQLite'taki dokuman
parcalariyla cosine similarity hesaplayarak en alakali baglami bulur ve
bu baglami Foundry Local uzerindeki LLM'e (orn. Phi-3.5) sistem prompt'u
ile besleyip yanit uretir.
"""

import json
import sqlite3

import numpy as np
from foundry_local import FoundryLocalManager
from openai import OpenAI

from ingest import CHAT_MODEL_ALIAS, DB_PATH, get_chunk_count, get_embedding

SYSTEM_PROMPT_TEMPLATE = """Sen, yalnizca kullaniciya saglanan BAGLAM icindeki bilgilere dayanarak \
soru cevaplayan bir doküman asistanisin.

Kurallar:
1. Yalnizca asagidaki BAGLAM icinde yer alan bilgileri kullanarak cevap ver.
2. Eger cevap BAGLAM icinde yoksa, kesinlikle uydurma veya tahmin yurutme; \
acikca "Bu bilgi saglanan dokumanlarda bulunmuyor." de.
3. Mumkun oldugunca hangi kaynaktan (dosya adindan) yararlandigini belirt.
4. Yanitlarini kisa, net ve anlasilir bir Turkce ile ver.

BAGLAM:
{context}
"""


class RAGEngine:
    def __init__(self, db_path: str = DB_PATH, chat_model_alias: str = CHAT_MODEL_ALIAS):
        self.db_path = db_path
        self._chat_manager = None
        self._chat_client = None
        self._chat_model_id = None
        self._init_chat_model(chat_model_alias)

    def _init_chat_model(self, alias: str) -> None:
        try:
            self._chat_manager = FoundryLocalManager(alias)
            model_info = self._chat_manager.get_model_info(alias)
            self._chat_model_id = model_info.id
            self._chat_client = OpenAI(
                base_url=self._chat_manager.endpoint,
                api_key=self._chat_manager.api_key,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Foundry Local sohbet modeli baslatilamadi ('{alias}'). Foundry Local "
                f"servisinin calistigindan ve modelin indirildiginden emin olun. Detay: {exc}"
            ) from exc

    def _load_all_chunks(self) -> list:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT source, content, embedding, page_number FROM chunks")
            return cursor.fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "Veritabani bulunamadi veya bos. Lutfen once dokumanlarinizi ice aktarin (ingest)."
            ) from exc
        finally:
            conn.close()

    @staticmethod
    def _format_source(source: str, page_number) -> str:
        return f"{source} (sayfa {page_number})" if page_number else source

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get_top_chunks(self, query: str, top_k: int = 3) -> list:
        """Sorguya en alakali top_k (skor, kaynak, icerik, sayfa_no) dortlusunu dondurur."""
        rows = self._load_all_chunks()
        if not rows:
            return []

        query_embedding = get_embedding(query)

        scored = []
        for source, content, embedding_json, page_number in rows:
            try:
                embedding = json.loads(embedding_json)
            except (json.JSONDecodeError, TypeError):
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((score, source, content, page_number))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def _build_system_prompt(self, top_chunks: list) -> str:
        if not top_chunks:
            context = "(Bu soru icin ilgili bir baglam bulunamadi.)"
        else:
            context = "\n\n".join(
                f"[Kaynak: {self._format_source(source, page_number)}]\n{content}"
                for _, source, content, page_number in top_chunks
            )
        return SYSTEM_PROMPT_TEMPLATE.format(context=context)

    def answer_query(self, question: str, top_k: int = 3) -> dict:
        """
        Kullanici sorusunu cevaplar.
        Donen sozluk: {"answer": str, "sources": list[str], "retrieved_chunks": list}
        """
        if not question or not question.strip():
            return {"answer": "Lutfen bir soru girin.", "sources": [], "retrieved_chunks": []}

        if get_chunk_count(self.db_path) == 0:
            return {
                "answer": (
                    "Veritabaninda henuz hicbir dokuman yok. Lutfen once dokuman yukleyip "
                    "'Ice Aktar' islemini calistirin."
                ),
                "sources": [],
                "retrieved_chunks": [],
            }

        try:
            top_chunks = self.get_top_chunks(question, top_k=top_k)
        except RuntimeError as exc:
            return {"answer": f"Hata: {exc}", "sources": [], "retrieved_chunks": []}

        system_prompt = self._build_system_prompt(top_chunks)

        try:
            response = self._chat_client.chat.completions.create(
                model=self._chat_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "Model bos bir yanit dondurdu. Lutfen sorunuzu farkli sekilde tekrar deneyin."
        except Exception as exc:
            return {
                "answer": f"Model yanit uretirken bir hata olustu: {exc}",
                "sources": [],
                "retrieved_chunks": top_chunks,
            }

        sources = sorted({self._format_source(source, page_number) for _, source, _, page_number in top_chunks})
        return {"answer": answer, "sources": sources, "retrieved_chunks": top_chunks}
