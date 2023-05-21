import os
import traceback
from utils.simple_logger import Log
import gradio as gr
import argparse
import sys
from pathlib import Path
from doc_search_llm.modules.chroma_processor import ChromaProcessor
from doc_search_llm.modules.model_processor import ModelProcessor
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from text2vec import SentenceModel


sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(args):

    try:
        # load the LLM model
        model, tokenizer = ModelProcessor.load_model(args)

        # create the LLM pipeline
        pipe = pipeline("text-generation", model=model,
                        tokenizer=tokenizer, max_new_tokens=1024)
        llm = HuggingFacePipeline(pipeline=pipe)
        # load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
    except Exception as e:
        log.error(f'Error loading model: {e}')
        raise e
    return chain


def load_args():
    # process command line arguments
    parser = argparse.ArgumentParser(description='Process the arguments.')
    parser.add_argument('--model_name_or_path', type=str,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=100, help='chunk overlap (default: 100)')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Load in 8 bits')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--top_n_docs_feed_llm', type=int,
                        default=4,  help='to avoid LLM too many documents, we only feed top N best matched documents to LLM')
    parser.add_argument('--port', type=int, default=7860, help='port number to listen on')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='server name to listen on')
    parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')
    parser.add_argument('--chatglm', action='store_true', help='Use chatglm')

    args = parser.parse_args()
    log.info(f'args: {args}')

    return args


def vectorstore_from_docs(vectorstore_processor, docs, embeddings=None):
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    if chunk_size > 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_docs = text_splitter.split_documents(docs)
        log.info(
            f'Your original documents have been splitted into {len(splitted_docs)} documents')
    try:
        vectorstore_processor.convert_from_docs(splitted_docs, embeddings=embeddings)
    except Exception as e:
        log.error(f'Error converting documents to vectorstore: {e}')
        raise e


def query_to_llm(vectorstore_processor, chain, query):
    if vectorstore_processor.vectorstore is not None:
        docs = vectorstore_processor.vectorstore.similarity_search(query, args.top_n_docs_feed_llm)
    else:
        log.error(f'Vectorstore is not loaded')
        raise Exception('Vectorstore is not loaded')

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

    # embeddings for chatGLM
    if args.chatglm:
        embeddings = SentenceModel('shibing624/text2vec-base-chinese')
    else:
        embeddings = None

    # show the file content in the text box
    # show the chatbot response in the text box
    with gr.Blocks() as demo:

        # file_input = gr.File(label="Upload File")
        # output = gr.Markdown()
        def process_file(file):
            with open(file.name, encoding="utf-8") as f:
                try:
                    content = f.read()
                    doc = Document(page_content=content, page_title=file.name,
                                   page_url=file.name, page_id=file.name)
                except Exception as e:
                    log.error(f'Error reading file: {e}')
                    log.error(traceback.format_exc())
                    return None

                log.debug(f'converted doc from upload file: {doc}')
                docs = [doc]
                vectorstore_from_docs(vectorstore_processor, docs, embeddings=embeddings)
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

    try:
        demo.launch(server_name=args.server_name if args.server_name is not None else '0.0.0.0',
                    server_port=args.port if args.port is not None else 5000)
    except Exception as e:
        log.error(f'Error launching the demo: {e}')
        log.error(f'server_name: {args.server_name}')
        log.error(f'port: {args.port}')
        log.error(traceback.format_exc())
        raise e
