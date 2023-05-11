import gradio as gr

def process_file(file):
    return f"File name: {file.name}, File type: {file.type}"

iface = gr.Interface(fn=process_file, inputs="file", outputs="text")

iface.launch()