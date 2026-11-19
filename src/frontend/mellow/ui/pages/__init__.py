from __future__ import annotations

import gradio as gr


PAGE_ORDER = ("home", "single", "batch", "workers", "feedback", "history")


def bind_pages(
    pages, controls, state, content, run_single, process_csv, submit_feedback,
    render_history, render_feedback,
):
    page_components = [pages[key] for key in PAGE_ORDER]
    nav_keys = ("home", "single", "batch", "workers", "feedback", "history")

    def navigate(index: int):
        updates = [gr.update(visible=position == index) for position in range(len(PAGE_ORDER))]
        # Keep active navigation styling synchronized with the visible page.
        nav_updates = [
            gr.update(elem_classes=["nav-btn", "active"] if position == index else ["nav-btn"])
            for position in range(len(nav_keys))
        ]
        return updates + nav_updates

    nav_components = [controls[f"nav_{key}"] for key in nav_keys]
    for index, key in enumerate(nav_keys):
        nav_components[index].click(
            lambda i=index: navigate(i),
            outputs=page_components + nav_components,
        )

    controls["home"]["single"].click(lambda: navigate(1), outputs=page_components + nav_components)
    controls["home"]["batch"].click(lambda: navigate(2), outputs=page_components + nav_components)

    single = controls["single"]
    single["run"].click(
        run_single,
        inputs=[single["text"], single["engine"], state["history"], gr.State(content)],
        outputs=[
            single["result"], state["history"], single["status"],
            state["latest_entry"], single["feedback_yes"], single["feedback_no"],
        ],
    ).then(
        lambda history: render_history(history, content),
        inputs=state["history"],
        outputs=controls["history"]["list"],
    )

    def handle_feedback(entry, correct, items):
        updated, status, lock_yes, lock_no, updated_entry = submit_feedback(
            entry, correct, items, content
        )
        return (
            updated, status, updated_entry,
            gr.update(interactive=not lock_yes),
            gr.update(interactive=not lock_no),
        )

    single["feedback_yes"].click(
        lambda entry, items: handle_feedback(entry, True, items),
        inputs=[state["latest_entry"], state["feedback"]],
        outputs=[
            state["feedback"], single["feedback_status"], state["latest_entry"],
            single["feedback_yes"], single["feedback_no"],
        ],
    ).then(
        lambda items: render_feedback(items, content),
        inputs=state["feedback"],
        outputs=controls["feedback"]["list"],
    )
    single["feedback_no"].click(
        lambda entry, items: handle_feedback(entry, False, items),
        inputs=[state["latest_entry"], state["feedback"]],
        outputs=[
            state["feedback"], single["feedback_status"], state["latest_entry"],
            single["feedback_yes"], single["feedback_no"],
        ],
    ).then(
        lambda items: render_feedback(items, content),
        inputs=state["feedback"],
        outputs=controls["feedback"]["list"],
    )

    batch = controls["batch"]
    batch["run"].click(
        process_csv,
        inputs=[batch["file"], batch["engine"], gr.State(content)],
        outputs=[batch["summary"], batch["download"], batch["preview"]],
    )

    history = controls["history"]
    history["clear"].click(
        lambda: ([], content["history"]["empty"]),
        outputs=[state["history"], history["list"]],
    )
