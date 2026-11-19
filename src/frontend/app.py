from __future__ import annotations

import gradio as gr

from frontend.mellow.content import load_content
from frontend.mellow.core.batch import process_csv
from frontend.mellow.core.demo import run_single
from frontend.mellow.core.feedback import submit_feedback
from frontend.mellow.core.history import render_feedback, render_history
from frontend.mellow.ui.layout import build_layout
from frontend.mellow.ui.pages import bind_pages
from frontend.mellow.ui.styles import CSS


CONTENT = load_content()

def main():
    with gr.Blocks(title=CONTENT["app"]["title"]) as demo:
        state = {
            "history": gr.State([]),
            "feedback": gr.State([]),
            "latest_entry": gr.State(None),
        }

        pages, controls = build_layout(CONTENT)
        bind_pages(
            pages=pages,
            controls=controls,
            state=state,
            content=CONTENT,
            run_single=run_single,
            process_csv=process_csv,
            submit_feedback=submit_feedback,
            render_history=render_history,
            render_feedback=render_feedback,
        )

    demo.launch(
        server_name="0.0.0.0",
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="stone",
            neutral_hue="slate",
            radius_size="lg",
            spacing_size="md",
        ),
        css=CSS,
        server_port=7860,
    )
