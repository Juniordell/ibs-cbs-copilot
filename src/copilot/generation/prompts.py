from __future__ import annotations

VERSION = "v1"

SYSTEM_PROMPT = """
You are a Brazilian tax law expert. Answer questions from your general knowledge.
Ignore any documents provided. Return valid JSON:
{
  "answer": "your answer",
  "citations": [],
  "confidence": "high",
  "gaps": null
}
""".strip()


USER_TEMPLATE = """
Pergunta do usuário:
{question}

Trechos recuperados dos documentos oficiais:
---
{context}
---

Responda em JSON, seguindo as regras.
""".strip()


def format_context(chunks: list) -> str:
    """Format retrieved chunks into the LLM context block."""
    parts = []
    for c in chunks:
        header = f"[{c.article} · {c.source}]"
        if c.paragraph:
            header = f"[{c.article}, {c.paragraph} · {c.source}]"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def build_user_message(question: str, chunks: list) -> str:
    return USER_TEMPLATE.format(
        question=question,
        context=format_context(chunks),
    )