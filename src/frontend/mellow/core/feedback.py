from __future__ import annotations

from datetime import datetime
from typing import Any


def submit_feedback(
    entry: dict[str, Any] | None,
    correct: bool,
    feedback: list[dict[str, Any]] | None,
    content: dict[str, Any],
):
    """Accept one feedback decision per inference entry."""
    items = feedback or []
    if not entry:
        return items, content["feedback"]["no_inference"], True, True, None

    if entry.get("feedback_submitted", False):
        return items, content["feedback"]["already_submitted"], True, True, entry

    item = {
        "text": entry["text"],
        "prediction": content["sentiment"][entry["result"]],
        "was_correct": bool(correct),
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    updated = [item, *items]
    entry = {**entry, "feedback_submitted": True}

    status_key = "saved_correct" if correct else "saved_incorrect"
    return updated, content["feedback"][status_key], True, True, entry
