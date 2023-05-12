from doc_search_llm.modules.model_processor import ModelProcessor
from doc_search_llm.modules.directory_processor import DirectoryProcessor
from doc_search_llm.modules.chroma_processor import ChromaProcessor


from utils.simple_logger import Log
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from aiohttp import web
import aiohttp

import argparse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)


async def websocket_handler(request):
    log.debug(f"New connection from client ip: {request.remote}")
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            query = msg.data
            log.debug(f"Received query: {query}")

            if query is None or len(query) == 0:
                await ws.send_str("No query provided")
                continue

            if len(query) < 10:
                await ws.send_str("Query too short")
                continue

            vectorstore_processor = request.app['vectorstore_processor']
            docs = vectorstore_processor.vectorstore.similarity_search(query)
            if docs is not None:
                log.info(f'best matched docs count: {len(docs)}')
                log.debug(f'Best matched docs: {docs[0]}')

                chain = request.app['chain']
                answer = chain.run(input_documents=docs, question=query)
                log.debug(f'Answer: {answer}')

                await ws.send_str(str(answer))
            else:
                await ws.send_str("No answer found in the knowledge base")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            log.error('WebSocket connection closed with exception %s' %
                      ws.exception())
            await ws.close()
            break

    log.info('WebSocket connection closed')
    return ws


async def on_startup(app):
    args = app['args']

    directory_processor = DirectoryProcessor(
        docs_root=args.docs_root, global_kwargs=args.global_kwargs)
    try:
        docs = directory_processor.load(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    except Exception as e:
        log.error(f'Error loading documents: {e}')
        raise e

    vectorstore_processor = ChromaProcessor()
    try:
        vectorstore_processor.convert_from_docs(docs)
        app['vectorstore_processor'] = vectorstore_processor

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
    app['chain'] = chain


async def on_cleanup(app):
    log.info('Cleaning up resources before shutdown')

if __name__ == "__main__":

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process the arguments.')
    parser.add_argument('--modle_name_or_path', type=str,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=100, help='chunk overlap (default: 100)')
    parser.add_argument('--docs_root', type=str,
                        required=True, help='docs root directory')
    parser.add_argument('--global_kwargs', type=str,
                        default="**/*.txt", help='global kwargs (default: **/*.txt')
    parser.add_argument('--persist_directory', type=str, default="chroma_storage",
                        help='persist directory (default: chroma_storage')
    parser.add_argument('--query', type=str, required=True,
                        help='query string, used to query against the docs')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Load in 8 bits')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--doc_count_for_qa', type=int,
                        default=4,  help='doc count for QA')
    parser.add_argument('--port', type=int, default=5000, help='port number')

    args = parser.parse_args()
    log.debug(f'args: {args}')

    # Create app and register on_startup function and route
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get('/ws', websocket_handler)

    # Store args in the app object
    app['args'] = args

    # Run the app
    web.run_app(app, port=args.port)
