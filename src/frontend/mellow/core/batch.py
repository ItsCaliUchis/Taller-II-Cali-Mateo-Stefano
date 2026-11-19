from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

from ..core.demo import CLASSES, demo_inference


def process_csv(file_path: str | None, engine: str, content: dict[str, Any]):
    if not file_path:
        return content["batch"]["missing_file"], None, ""

    source = Path(file_path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError(content["batch"]["missing_headers"])
            candidates = [
                name for name in reader.fieldnames
                if name.lower() in {"text", "review", "review_text", "content"}
            ]
            text_column = candidates[0] if candidates else reader.fieldnames[0]
            rows = list(reader)

        output_rows = []
        counts = {key: 0 for key in CLASSES}
        for row in rows:
            review = (row.get(text_column) or "").strip()
            if not review:
                continue
            probabilities, result = demo_inference(review)
            counts[result] += 1
            output_rows.append({
                **row,
                "sentiment_demo": content["sentiment"][result],
                "negative_probability": round(probabilities["negative"], 5),
                "neutral_probability": round(probabilities["neutral"], 5),
                "positive_probability": round(probabilities["positive"], 5),
            })

        if not output_rows:
            raise ValueError(content["batch"]["no_reviews"].format(column=text_column))

        output = tempfile.NamedTemporaryFile(
            mode="w", suffix="_mellow_demo.csv", prefix="mellow_",
            delete=False, newline="", encoding="utf-8-sig",
        )
        with output:
            writer = csv.DictWriter(output, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)

        summary = content["batch"]["summary"].format(
            total=len(output_rows),
            positive=counts["positive"],
            neutral=counts["neutral"],
            negative=counts["negative"],
            column=text_column,
            engine=engine,
        )
        preview_lines = [
            f"| {str(row.get(text_column, ''))[:72].replace('|', ' ')} | {row['sentiment_demo']} |"
            for row in output_rows[:8]
        ]
        preview = (
            f"| {content['batch']['preview_review']} | {content['batch']['preview_sentiment']} |\n"
            "|---|---|\n" + "\n".join(preview_lines)
        )
        return summary, output.name, preview

    except Exception as error:
        return content["batch"]["error"].format(error=str(error)), None, ""
