from __future__ import annotations

from datetime import datetime
from typing import Any

from ..content import text

POSITIVE_WORDS = {
    "good", "great", "love", "excellent", "amazing", "perfect", "beautiful",
    "easy", "fast", "recommend", "happy", "wonderful", "best", "nice",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "broken", "poor", "slow", "waste",
    "disappointed", "worst", "cheap", "problem", "defective",
}
CLASSES = ("negative", "neutral", "positive")


def demo_inference(review: str) -> tuple[dict[str, float], str]:
    """Lexical demo only. This is not a trained inference model."""
    tokens = [
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in review.split()
    ]
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    neutral = max(1, len(tokens) * 0.10)

    raw = {
        "negative": 1.0 + negative * 2.2,
        "neutral": neutral + (1.0 if positive == negative else 0.0),
        "positive": 1.0 + positive * 2.2,
    }
    total = sum(raw.values())
    probabilities = {key: value / total for key, value in raw.items()}
    return probabilities, max(probabilities, key=probabilities.get)


def probability_panel(probabilities: dict[str, float], result: str, content: dict[str, Any]) -> str:
    labels = content["sentiment"]
    rows = []
    for key in CLASSES:
        percentage = probabilities[key] * 100
        rows.append(
            f"""
            <div class="prob-row">
              <div class="prob-meta"><span>{labels[key]}</span><b>{percentage:.1f}%</b></div>
              <div class="prob-track"><span class="prob-fill {key}" style="width:{percentage:.1f}%"></span></div>
            </div>
            """
        )
    return f"""
    <div class="result-card juicy-enter">
      <div class="result-topline">
        <span class="eyebrow">{content['result']['estimated']}</span>
        <span class="demo-tag">{content['result']['demo_tag']}</span>
      </div>
      <div class="result-verdict {result}">
        <span class="verdict-dot"></span>
        <div><small>{content['result']['detected']}</small><strong>{labels[result]}</strong></div>
      </div>
      <div class="prob-title">{content['result']['distribution']}</div>
      {''.join(rows)}
      <p class="result-note">{content['result']['demo_note']}</p>
    </div>
    """


def run_single(review: str, engine: str, history: list[dict[str, Any]] | None, content: dict[str, Any]):
    if not review or not review.strip():
        return (
            f"<div class='empty-result'>{content['inference']['empty_result']}</div>",
            history or [],
            content["inference"]["empty_status"],
            None,
            False,
            False,
        )

    probabilities, result = demo_inference(review)
    entry = {
        "text": review.strip(),
        "engine": engine,
        "result": result,
        "probabilities": probabilities,
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "feedback_submitted": False,
    }
    updated_history = [entry, *(history or [])][:30]
    status = content["inference"]["demo_status"].format(
        label=content["sentiment"][result],
        engine=engine,
    )
    return (
        probability_panel(probabilities, result, content),
        updated_history,
        status,
        entry,
        True,
        True,
    )
