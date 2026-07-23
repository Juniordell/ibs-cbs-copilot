import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
from openai import OpenAI

from src.copilot.retrieval.vector import VectorRetriever
from src.copilot.retrieval.bm25 import BM25Retriever
from src.copilot.retrieval.hybrid import HybridRetriever

load_dotenv()

QUERIES = [
    "Qual a alíquota do IBS?",
    "Quem é contribuinte da CBS?",
    "Como funciona o split payment?",
    "O que é cashback tributário?",
    "Base de cálculo do IBS em operações com desconto",
]


def main():
    db_url = os.environ["DATABASE_URL"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    vec = VectorRetriever(db_url, client)
    bm25 = BM25Retriever(db_url)
    hybrid = HybridRetriever(db_url, client)

    for q in QUERIES:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("=" * 80)

        for name, retriever in [("VECTOR", vec), ("BM25", bm25), ("HYBRID", hybrid)]:
            print(f"\n--- {name} ---")
            for i, r in enumerate(retriever.retrieve(q, k=5), 1):
                print(f"[{i}] {r.score:.3f} | {r.article} · {r.source}")


if __name__ == "__main__":
    main()