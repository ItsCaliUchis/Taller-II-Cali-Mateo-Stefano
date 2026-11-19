from __future__ import annotations

from typing import Any


def render_history(history: list[dict[str, Any]] | None, content: dict[str, Any]) -> str:
    if not history:
        return content["history"]["empty"]
    cards = []
    for item in history[:12]:
        snippet = _escape(item["text"])
        cards.append(
            f"""<div class="history-item">
              <div><span class="history-label">{content['sentiment'][item['result']]}</span>
              <small>{item['created_at']} · {item['engine']}</small></div>
              <p>{snippet[:180]}{'…' if len(snippet) > 180 else ''}</p>
            </div>"""
        )
    return "<div class='history-list'>" + "".join(cards) + "</div>"


def render_feedback(feedback: list[dict[str, Any]] | None, content: dict[str, Any]) -> str:
    if not feedback:
        return content["feedback"]["empty_list"]
    cards = []
    for item in feedback[:15]:
        correct = item["was_correct"]
        status = content["feedback"]["correct"] if correct else content["feedback"]["incorrect"]
        css_class = "positive" if correct else "negative"
        snippet = _escape(item["text"])
        cards.append(
            f"""<div class="history-item">
              <div><span class="history-label {css_class}">{status}</span>
              <small>{item['created_at']}</small></div>
              <p>{snippet[:160]}</p>
            </div>"""
        )
    return "<div class='history-list'>" + "".join(cards) + "</div>"


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
