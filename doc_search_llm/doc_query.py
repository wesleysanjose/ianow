from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import argparse

import sys
from pathlib import Path
from doc_search_llm.directory_processor import DirectoryProcessor
from doc_search_llm.vectorstore_processor import VectorstoreProcessor
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log.get_logger(__name__)

def main(args):
    directory_processor = DirectoryProcessor(docs_root=args.docs_root, global_kwargs=args.global_kwargs)
    docs = directory_processor.load(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    #embeddings = HuggingFaceEmbeddings(model_name=args.modle_name_or_path)
    vectorstore_processor = VectorstoreProcessor()
    vectorstore_processor.convert_from_docs(docs)

    if args.query is not None:
        # search the query through similarit search against documents
        query = args.query
        results = vectorstore_processor.vectorstore.similarity_search(query)

        if results is not None:
            log.info(f'matched docs: {len(results)}')
            for i, result in enumerate(results):
                log.info(f'matched doc {i}: {result}')

        # search the query through LLM
        tokenizer = AutoTokenizer.from_pretrained(args.modle_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.modle_name_or_path, device_map='auto', load_in_8bit=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
        llm = HuggingFacePipeline(pipeline=pipe)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = vectorstore_processor.vectorstore.similarity_search(query, 3, include_metadata=True)
        chain.run(input_documents=docs, question=query)

if __name__ == "__main__":

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--modle_name_or_path', type=str, help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int, default=0, help='chunk overlap (default: 0)')
    parser.add_argument('--docs_root', type=str, required=True, help='docs root directory')
    parser.add_argument('--global_kwargs', type=str, default="**/*.txt", help='global kwargs (default: **/*.txt')
    parser.add_argument('--persist_directory', type=str, default="chroma_storage", help='persist directory (default: chroma_storage')
    parser.add_argument('--query', type=str, required=True, help='query string, used to query against the docs')
    args = parser.parse_args()
    log.info(f'args: {args}')

    main(args)

    
