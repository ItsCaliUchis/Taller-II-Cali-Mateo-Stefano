from __future__ import annotations

import gradio as gr


def build_layout(content: dict):
    controls = {}
    pages = {}

    with gr.Column(elem_id="app-shell"):
        gr.HTML(content["html"]["topbar"])

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=205, elem_id="sidebar"):
                gr.HTML(content["html"]["nav_workspace"])
                controls["nav_home"] = gr.Button(content["nav"]["home"], elem_classes=["nav-btn", "active"])
                controls["nav_single"] = gr.Button(content["nav"]["single"], elem_classes="nav-btn")
                controls["nav_batch"] = gr.Button(content["nav"]["batch"], elem_classes="nav-btn")
                gr.HTML(content["html"]["nav_tracking"])
                controls["nav_workers"] = gr.Button(content["nav"]["workers"], elem_classes="nav-btn")
                controls["nav_feedback"] = gr.Button(content["nav"]["feedback"], elem_classes="nav-btn")
                controls["nav_history"] = gr.Button(content["nav"]["history"], elem_classes="nav-btn")
                gr.HTML(content["html"]["sidebar_note"])

            with gr.Column(scale=4, elem_id="page-content"):
                from .pages.batch import build_batch_page
                from .pages.feedback import build_feedback_page
                from .pages.history import build_history_page
                from .pages.home import build_home_page
                from .pages.inference import build_inference_page
                from .pages.workers import build_workers_page

                pages["home"], controls["home"] = build_home_page(content)
                pages["single"], controls["single"] = build_inference_page(content)
                pages["batch"], controls["batch"] = build_batch_page(content)
                pages["workers"], controls["workers"] = build_workers_page(content)
                pages["feedback"], controls["feedback"] = build_feedback_page(content)
                pages["history"], controls["history"] = build_history_page(content)

    return pages, controls
