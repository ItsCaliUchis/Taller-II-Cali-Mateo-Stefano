import gradio as gr


def build_batch_page(content):
    with gr.Column(visible=False) as page:
        gr.HTML(content["html"]["batch_heading"])
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes="soft-card"):
                    gr.HTML(content["html"]["batch_card_heading"])
                    gr.HTML(content["html"]["english_notice"])
                    file = gr.File(
                        label=content["batch"]["file_label"],
                        file_types=[".csv"],
                        type="filepath",
                    )
                    engine = gr.Dropdown(
                        choices=content["engines"],
                        value=content["engines"][0],
                        label=content["batch"]["engine_label"],
                        elem_id="batch-engine-select",
                    )
                    run = gr.Button(content["batch"]["run_button"], variant="primary")
            with gr.Column(scale=3):
                summary = gr.HTML(content["html"]["batch_placeholder"])
                preview = gr.Markdown()
                download = gr.File(label=content["batch"]["download_label"])
    return page, {
        "file": file, "engine": engine, "run": run,
        "summary": summary, "preview": preview, "download": download,
    }
