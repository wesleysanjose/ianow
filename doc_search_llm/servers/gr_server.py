import gradio as gr
import argparse
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

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process the arguments.')
    parser.add_argument('--modle_name_or_path', type=str,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=100, help='chunk overlap (default: 100)')
    parser.add_argument('--global_kwargs', type=str,
                        default="**/*.txt", help='global kwargs (default: **/*.txt')
    parser.add_argument('--persist_directory', type=str, default="chroma_storage",
                        help='persist directory (default: chroma_storage')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Load in 8 bits')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--doc_count_for_qa', type=int,
                        default=4,  help='doc count for QA')
    parser.add_argument('--port', type=int, default=5000, help='port number')

    args = parser.parse_args()
    log.info(f'args: {args}')

    demo.launch(server_name="0.0.0.0")
