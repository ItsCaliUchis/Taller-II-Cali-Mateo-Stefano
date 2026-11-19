import gradio as gr


def build_workers_page(content):
    with gr.Column(visible=False) as page:
        gr.HTML(content["html"]["workers_heading"])
        gr.HTML(content["html"]["workers_cards"])
        gr.HTML(content["html"]["workers_note"])
    return page, {}
