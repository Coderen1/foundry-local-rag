"""
evaluate.py
-----------
Kucuk bir soru-cevap test setini (bkz. test_questions.example.json) RAGEngine
uzerinde calistirip her sorunun "gectiini/kaldigini" raporlar. Orijinal proje
planinin Hafta 5 "Functional Testing" adimini (cevaplanabilir + cevaplanamaz
sorularin sistematik olarak denenmesi) otomatiklestirir.

Kullanim:
    cp test_questions.example.json test_questions.json
    # test_questions.json dosyasini kendi dokumanlariniza gore duzenleyin
    python evaluate.py
    python evaluate.py baska_test_dosyasi.json
"""

import argparse
import json
import sys
from pathlib import Path

from rag_engine import RAGEngine

DEFAULT_TEST_FILE = "test_questions.json"

# Modelin "bu bilgi dokumanlarda yok" derken kullanmasi beklenen ifadeler
# (rag_engine.SYSTEM_PROMPT_TEMPLATE'teki talimatla uyumlu).
NOT_FOUND_MARKERS = ["bulunmuyor", "bilmiyorum", "bilgi yok", "bilgim yok"]


def load_test_cases(test_file: str) -> list:
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(
            f"'{test_file}' bulunamadi. Once 'test_questions.example.json' dosyasini "
            f"'{DEFAULT_TEST_FILE}' olarak kopyalayip kendi dokumanlariniza gore "
            f"duzenleyin."
        )
    try:
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{test_file}' gecerli bir JSON dosyasi degil: {exc}") from exc

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"'{test_file}' bos veya beklenen formatta degil (bir JSON listesi olmali).")
    return cases


def evaluate_case(engine: RAGEngine, case: dict, top_k: int) -> dict:
    question = case["question"]
    expect_answerable = case.get("expect_answerable", True)
    expected_keywords = case.get("expected_keywords", [])

    result = engine.answer_query(question, top_k=top_k)
    answer_lower = result["answer"].lower()

    if expect_answerable:
        passed = all(keyword.lower() in answer_lower for keyword in expected_keywords)
    else:
        passed = any(marker in answer_lower for marker in NOT_FOUND_MARKERS)

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "passed": passed,
    }


def run_evaluation(test_file: str, top_k: int = 3) -> list:
    cases = load_test_cases(test_file)
    engine = RAGEngine()
    return [evaluate_case(engine, case, top_k) for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG asistanini bir test soru setine karsi calistirir.")
    parser.add_argument("test_file", nargs="?", default=DEFAULT_TEST_FILE, help="Test sorularini iceren JSON dosyasi.")
    parser.add_argument("--top-k", type=int, default=3, help="Sorgu basina getirilecek parca sayisi.")
    args = parser.parse_args()

    try:
        results = run_evaluation(args.test_file, top_k=args.top_k)
    except Exception as exc:
        print(f"[HATA] {exc}")
        raise SystemExit(1)

    passed_count = 0
    for result in results:
        status = "GECTI" if result["passed"] else "KALDI"
        passed_count += result["passed"]
        print(f"[{status}] {result['question']}")
        print(f"  Yanit: {result['answer'][:200]}")
        print(f"  Kaynaklar: {result['sources']}")
        print()

    total = len(results)
    print(f"Sonuc: {passed_count}/{total} test gecti.")
    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
