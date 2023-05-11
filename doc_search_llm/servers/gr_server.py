import gradio as gr
from .arg_parser import load_parser

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log.get_logger(__name__)

# show the file content in the text box
# show the chatbot response in the text box
with gr.Blocks() as demo:


    # file_input = gr.File(label="Upload File")
    # output = gr.Markdown()
    def process_file(file):
        with open(file.name, encoding="utf-8") as f:
            content = f.read()
            return content
    
    gr.Interface(fn=process_file, inputs="file", outputs="text")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = "hello"
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":

    args = load_parser()
    log.info(f'args: {args}')

    demo.launch(server_name="0.0.0.0")
