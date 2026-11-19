import gradio as gr


def build_home_page(content):
    with gr.Column(visible=True) as page:
        gr.HTML(content["html"]["home_hero"])
        gr.HTML(content["html"]["home_heading"])
        with gr.Row():
            for stat in content["home"]["stats"]:
                with gr.Column():
                    gr.HTML(content["html"]["stat_card"].format(**stat))
        with gr.Row():
            with gr.Column():
                gr.HTML(content["html"]["home_single_card"])
                go_single = gr.Button(content["home"]["single_button"], variant="primary")
            with gr.Column():
                gr.HTML(content["html"]["home_batch_card"])
                go_batch = gr.Button(content["home"]["batch_button"], variant="secondary")
        gr.HTML(content["html"]["footer"])
    return page, {"single": go_single, "batch": go_batch}
