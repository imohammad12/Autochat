import argparse
import gradio as gr
from typing import List, Tuple
from transformers import HfArgumentParser
from src.auto_chat.pipline import Pipeline
from src.auto_chat.utils import add_input_args
from src.auto_chat.arguments import PipelineArguments, EmbeddingModelArguments, ServerArguments
from src.auto_chat.chatbot_utils import EXAMPLES, TITLE_MARKDOWN, get_metadata_html


def clear_history():
    pipeline.chat_history.clear()
    return "", [(None, None)], None


def insert_input(user_input: str, chat_history: List[Tuple[str, str]]):
    response = pipeline(user_input)
    chat_history.append((user_input, response['output']))
    metadata = get_metadata_html(pipeline.ir_queries[-1], pipeline.ir_outputs[-1])
    return "", chat_history, metadata


def build_demo():
    with gr.Blocks(title="AutoChat") as demo:
        state = gr.State()

        gr.Markdown(TITLE_MARKDOWN)

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbot", visible=True, height=580)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            visible=True,
                            container=False
                        )
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=True)
                with gr.Row(visible=True) as button_row:
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        gr_examples = gr.Examples(examples=EXAMPLES, inputs=[textbox])

        with gr.Accordion("Metadata", open=False, visible=True):
            conversation_metadata = gr.HTML(label="Conversation Details")

        ####### Button Actions #######
        clear_btn.click(
            clear_history,
            None,
            [textbox, chatbot, conversation_metadata]
        )

        textbox.submit(
            insert_input,
            [textbox, chatbot],
            [textbox, chatbot, conversation_metadata]
        )

        submit_btn.click(
            insert_input,
            [textbox, chatbot],
            [textbox, chatbot, conversation_metadata]
        )

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    hf_parser = HfArgumentParser([
        PipelineArguments,
        EmbeddingModelArguments,
        ServerArguments
    ])

    add_input_args(parser)
    config_file_path = parser.parse_args().config_file_path
    pipline_args, embed_args, server_args = hf_parser.parse_json_file(config_file_path)
    pipline_args: PipelineArguments
    embed_args: EmbeddingModelArguments
    server_args: ServerArguments

    pipeline = Pipeline(pipeline_args=pipline_args, embed_args=embed_args)

    gradio_demo = build_demo()
    gradio_demo.queue(
        api_open=False
    ).launch(
        server_name=server_args.server_ip,
        server_port=server_args.server_port,
        share=True
    )
