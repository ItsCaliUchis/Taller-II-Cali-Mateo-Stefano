import json
from pathlib import Path
from typing import Any


CONTENT_PATH = Path(__file__).resolve().parent.parent / "content" / "es.json"


def load_content(path: Path = CONTENT_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def text(content: dict[str, Any], key: str, **values: Any) -> str:
    """Resolve dotted JSON keys and interpolate named values."""
    current: Any = content
    for part in key.split("."):
        current = current[part]
    return str(current).format(**values)
