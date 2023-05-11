from utils.simple_logger import Log
import gradio as gr
import argparse
import sys
from pathlib import Path
from doc_search_llm.modules.chroma_processor import ChromaProcessor
from doc_search_llm.modules.directory_processor import DirectoryProcessor
from doc_search_llm.modules.model_processor import ModelProcessor
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)


def load_model(args):
    # load the LLM model
    model, tokenizer = ModelProcessor.load_model(args)

    # create the LLM pipeline
    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, max_new_tokens=1024)
    llm = HuggingFacePipeline(pipeline=pipe)
    # load the QA chain
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def load_args():
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

    return args

def vectorstore_from_docs(vectorstore_processor, docs):
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    if chunk_size > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splitted_docs = text_splitter.split_documents(docs)
            log.info(f'Your original documents have been splitted into {len(splitted_docs)} documents')
    vectorstore_processor.convert_from_docs(splitted_docs)

def query_to_llm(vectorstore_processor, chain, query):
    docs = vectorstore_processor.vectorstore.similarity_search(query)
    if docs is not None:
        log.info(f'best matched docs count: {len(docs)}')
        log.debug(f'Best matched docs: {docs[0]}')

        answer = chain.run(input_documents=docs, question=query)
        log.debug(f'answer: {answer}')
    return answer


if __name__ == "__main__":

    args = load_args()

    chain = load_model(args)

    vectorstore_processor = ChromaProcessor()

    # show the file content in the text box
    # show the chatbot response in the text box
    with gr.Blocks() as demo:

        # file_input = gr.File(label="Upload File")
        # output = gr.Markdown()
        def process_file(file):
            with open(file.name, encoding="utf-8") as f:
                content = f.read()
                doc = Document(page_content=content, page_title=file.name, page_url=file.name, page_id=file.name)
                log.debug(f'converted doc from upload file: {doc}')
                docs = [doc]
                vectorstore_from_docs(vectorstore_processor, docs)
                return content

        gr.Interface(fn=process_file, inputs="file", outputs="text")

        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            bot_message = query_to_llm(vectorstore_processor, chain, message)
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(server_name="0.0.0.0")
