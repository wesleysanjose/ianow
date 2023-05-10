from doc_search_llm.processors.model_processor import ModelProcessor
from doc_search_llm.processors.directory_processor import DirectoryProcessor
from doc_search_llm.processors.vectorstore_processor import VectorstoreProcessor

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

chain = None
docs = None


async def websocket_handler(request):
    log.info("New connection")
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            query = msg.data
            log.debug(f"Received query: {query}")
            vectorstore_processor = request.app['vectorstore_processor']
            docs = vectorstore_processor.vectorstore.similarity_search(query)
            if docs is not None:
                log.info(f'best matched docs count: {len(docs)}')
                log.debug(f'Best matched docs: {docs[0]}')

                chain = request.app['chain']
                answer = chain.run(input_documents=docs, question=query)

                await ws.send_str(str(answer))
            else:
                await ws.send_str("No answer found in the knowledge base")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            log.error('WebSocket connection closed with exception %s' %
                      ws.exception())

    log.info('WebSocket connection closed')
    return ws


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            log.info(f'query: {msg.data}')
            chain = request.app['chain']
            result = chain.run(input_documents=docs, question=query)

            await ws.send_str(chain.run(input_documents=docs, question=msg.data))
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('WebSocket connection closed with exception %s' %
                  ws.exception())

    print('WebSocket connection closed')
    return ws

app = web.Application()
app.router.add_get('/ws', websocket_handler)


async def on_startup(app):
    args = app['args']

    directory_processor = DirectoryProcessor(
        docs_root=args.docs_root, global_kwargs=args.global_kwargs)
    docs = directory_processor.load(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    vectorstore_processor = VectorstoreProcessor()
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
    app['chain'] = chain

if __name__ == "__main__":

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
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
    parser.add_argument('--llama', action='store_true', help='Enable the flag')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Load in 8 bits')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--doc_count_for_qa', type=int,
                        default=4,  help='doc count for QA')
    args = parser.parse_args()
    log.info(f'args: {args}')

    # Create app and register on_startup function and route
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get('/ws', websocket_handler)

    # Store args in the app object
    app['args'] = args

    # Run the app
    web.run_app(app, port=5000)
