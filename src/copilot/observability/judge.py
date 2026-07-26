from __future__ import annotations

import json
import logging
import os
import random

from anthropic import AsyncAnthropic
from langfuse import Langfuse

logger = logging.getLogger(__name__)

SAMPLE_RATE = 0.10  # judge 10% of traffic
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheap judge


JUDGE_PROMPT = """You are an evaluator. Score the answer's faithfulness to the contexts.

Contexts:
{contexts}

Question: {question}

Answer: {answer}

Does the answer contain ONLY statements supported by the contexts? Score 0.0 (fully unsupported) to 1.0 (fully supported).

Return ONLY valid JSON:
{{"score": 0.X, "reason": "one sentence"}}"""


def should_sample() -> bool:
    return random.random() < SAMPLE_RATE


async def judge_faithfulness(
    trace_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    langfuse: Langfuse,
    anthropic: AsyncAnthropic,
) -> None:
    """Run LLM-as-judge and post the score to the trace. Silent on failure."""
    try:
        message = await anthropic.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    contexts="\n\n".join(contexts[:5]),  # cap for cost
                    question=question,
                    answer=answer,
                ),
            }],
        )
        raw = message.content[0].text.strip()
        # Strip fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        data = json.loads(raw)

        langfuse.create_score(
            trace_id=trace_id,
            name="faithfulness_online",
            value=float(data["score"]),
            comment=data.get("reason", ""),
        )
        logger.info("Judged trace %s: %.2f", trace_id, data["score"])
    except Exception:
        logger.exception("Judge failed for trace %s", trace_id)