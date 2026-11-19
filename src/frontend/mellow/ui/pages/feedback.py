import gradio as gr


def build_feedback_page(content):
    with gr.Column(visible=False) as page:
        gr.HTML(content["html"]["feedback_heading"])
        listing = gr.HTML(content["feedback"]["empty_list"])
    return page, {"list": listing}
