import gradio as gr


def build_history_page(content):
    with gr.Column(visible=False) as page:
        gr.HTML(content["html"]["history_heading"])
        listing = gr.HTML(content["history"]["empty"])
        clear = gr.Button(content["history"]["clear_button"], variant="secondary")
    return page, {"list": listing, "clear": clear}
