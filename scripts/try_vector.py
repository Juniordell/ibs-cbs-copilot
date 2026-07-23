from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
from openai import OpenAI

from src.copilot.retrieval.vector import VectorRetriever

load_dotenv()

QUERIES = [
    "Qual a alíquota do IBS?",
    "Quem é contribuinte da CBS?",
    "Como funciona o split payment?",
    "O que é cashback tributário?",
    "Base de cálculo do IBS em operações com desconto",
]


def main():
    retriever = VectorRetriever(
        db_url=os.environ["DATABASE_URL"],
        openai_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    )

    for q in QUERIES:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("=" * 80)
        results = retriever.retrieve(q, k=5)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r.score:.3f} | {r.article} · {r.source}")
            print(f"    {r.text[:180]}...")


if __name__ == "__main__":
    main()