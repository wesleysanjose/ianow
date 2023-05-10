from doc_search_llm.processors.model_processor import ModelProcessor
from doc_search_llm.processors.directory_processor import DirectoryProcessor
from doc_search_llm.processors.vectorstore_processor import VectorstoreProcessor

from utils.simple_logger import Log
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import argparse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)


def main(args):
    directory_processor = DirectoryProcessor(
        docs_root=args.docs_root, global_kwargs=args.global_kwargs)
    docs = directory_processor.load(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # embeddings = HuggingFaceEmbeddings(model_name=args.modle_name_or_path)
    vectorstore_processor = VectorstoreProcessor()
    vectorstore_processor.convert_from_docs(docs)

    if args.query is not None:
        # search the query through similarit search against documents
        query = args.query

        model, tokenizer = ModelProcessor.load_model(args)

        pipe = pipeline("text-generation", model=model,
                        tokenizer=tokenizer, max_new_tokens=1024)
        llm = HuggingFacePipeline(pipeline=pipe)

        # load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # search the best matched documents
        docs = vectorstore_processor.vectorstore.similarity_search(
            query, args.doc_count_for_qa, include_metadata=True)
        log.info(f'similarity searched {len(docs)} documents')

        # run the LLM query by feeding the best matched documents
        result = chain.run(input_documents=docs, question=query)
        log.info(f'LLM query: {query}')
        log.info(f'LLM result: {result}')
    else:
        log.error(f'no query string provided, please provide a query string by using --query')

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
    parser.add_argument('--doc_count_for_qa', type=int, default=4,  help='doc count for QA')
    args = parser.parse_args()
    log.info(f'args: {args}')

    main(args)
