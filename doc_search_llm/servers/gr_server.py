import gradio as gr

def process_file(file):
    return f"File name: {file.name}, File type: {file.type}"

iface = gr.Interface(fn=process_file, inputs="file", outputs="text")

iface.launch(share=True, server_name="0.0.0.0")  # share=True will generate a link for you to share with others))