from langchain import LLMChain, PromptTemplate
from doc_search_llm.modules.model_processor import ModelProcessor
from doc_search_llm.modules.directory_processor import DirectoryProcessor
from doc_search_llm.modules.chroma_processor import ChromaProcessor

from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import argparse

from utils.simple_logger import Log
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    log.debug(f'args: {args}')

    # load the documents from the docs directory
    directory_processor = DirectoryProcessor(
        docs_root=args.docs_root, kwargs=args.kwargs)
    chroma_processor = ChromaProcessor()
    
    try:
        docs = directory_processor.load(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    except Exception as e:
        log.error(f'Error converting documents to vectorstore: {e}')
        raise e

    if args.query is not None:
        # get the query string
        query = args.query

        # load the LLM model
        try:
            embeddings = None
            if args.llamacpp:
                template = """Question: {question}
                    Answer:
                    """

                prompt = PromptTemplate(template=template, input_variables=["question"])
                model, embeddings = ModelProcessor.load_llamacpp(args)
                chain = LLMChain(prompt=prompt, llm=model)
            elif args.gpt4all:
                template = """Question: {question}
                    Answer: Let's think step by step."""

                prompt = PromptTemplate(template=template, input_variables=["question"])
                model = ModelProcessor.load_llamacpp(args)
                chain = LLMChain(prompt=prompt, llm=model)
            else:
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
        
        # convert the documents to vectorstore
        chroma_processor = ChromaProcessor()
        chroma_processor.convert_from_docs(docs, embeddings)

        # search top N best matched documents to reduce the scope
        top_matched_docs = chroma_processor.vectorstore.similarity_search(
            query, args.top_n_docs_feed_llm, include_metadata=True)
        log.info(f'similarity found {len(top_matched_docs)} documents can be feed to LLM')

        # run the LLM query by feeding the best matched documents
        result = chain.run(input_documents=top_matched_docs, question=query)
        log.info(f'LLM query: {query}')
        log.info(f'LLM result: {result}')
    else:
        log.error(
            f'no query string provided, please provide a query string by using --query')


if __name__ == "__main__":

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name_or_path', type=str,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=100, help='chunk overlap (default: 100)')
    parser.add_argument('--docs_root', type=str,
                        required=True, help='docs root directory')
    parser.add_argument('--kwargs', type=str,
                        default="**/*.txt", help='global kwargs (default: **/*.txt')
    parser.add_argument('--query', type=str, required=True,
                        help='query string, used to query against LLM with the context of loaded documents')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Use 8 bits to load the model')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bf16 to load the model if device supports it, otherwise use fp16')
    parser.add_argument('--top_n_docs_feed_llm', type=int,
                        default=4,  help='to avoid LLM too many documents, we only feed top N best matched documents to LLM')
    parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')
    parser.add_argument('--llamacpp', action='store_true', help='Use llamacpp')
    parser.add_argument('--gpt4all', action='store_true',  help='Use gpt4all')
    args = parser.parse_args()

    main(args)
