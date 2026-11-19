import gradio as gr


def build_inference_page(content):
    with gr.Column(visible=False) as page:
        gr.HTML(content["html"]["inference_heading"])
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="soft-card"):
                    gr.HTML(content["html"]["inference_card_heading"])
                    gr.HTML(content["html"]["english_notice"])
                    review = gr.Textbox(
                        label=content["inference"]["input_label"],
                        lines=8,
                        placeholder=content["inference"]["placeholder"],
                    )
                    engine = gr.Dropdown(
                        choices=content["engines"],
                        value=content["engines"][0],
                        label=content["inference"]["engine_label"],
                        elem_id="engine-select",
                    )
                    run = gr.Button(content["inference"]["run_button"], variant="primary")
                    status = gr.Markdown()
                gr.HTML(content["html"]["feedback_prompt"])
                with gr.Row():
                    yes = gr.Button(content["feedback"]["yes_button"], variant="secondary", interactive=False)
                    no = gr.Button(content["feedback"]["no_button"], variant="secondary", interactive=False)
                feedback_status = gr.Markdown()
            with gr.Column(scale=2):
                result = gr.HTML(content["html"]["result_placeholder"])
    return page, {
        "text": review, "engine": engine, "run": run, "status": status,
        "feedback_yes": yes, "feedback_no": no,
        "feedback_status": feedback_status, "result": result,
    }
