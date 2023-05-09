from utils.simple_logger import Log
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch

import argparse

import sys
from pathlib import Path
from doc_search_llm.directory_processor import DirectoryProcessor
from doc_search_llm.vectorstore_processor import VectorstoreProcessor
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
        # search the query through LLM

        # load the LLM model
        if torch.has_mps:
            log.info("Using MPS")
            device = torch.device('mps')
            model = model.to(device)
        else:
            log.info("Using CUDA")
            model = model.cuda()
            
        if args.load_in_8bit:
            log.info("Model Loading in 8 bits")
            model = AutoModelForCausalLM.from_pretrained(args.modle_name_or_path,
                                                         low_cpu_mem_usage=True,
                                                         device_map="auto",
                                                         quantization_config=BitsAndBytesConfig(
                                                             load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
                                                         trust_remote_code=True)
        else:
            log.info("Default model loading")
            model = AutoModelForCausalLM.from_pretrained(args.modle_name_or_path,
                                                         low_cpu_mem_usage=True,
                                                         torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                                                         trust_remote_code=True)

        if type(model) is LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.modle_name_or_path, clean_up_tokenization_spaces=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.modle_name_or_path, clean_up_tokenization_spaces=True)
        pipe = pipeline("text-generation", model=model,
                        tokenizer=tokenizer, max_new_tokens=1024)
        llm = HuggingFacePipeline(pipeline=pipe)

        # load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # search the best matched documents
        docs = vectorstore_processor.vectorstore.similarity_search(
            query, 10, include_metadata=True)

        # run the LLM query by feeding the best matched documents
        result = chain.run(input_documents=docs, question=query)
        log.info(f'LLM query: {query}')
        log.info(f'LLM result: {result}')


if __name__ == "__main__":

    # process command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--modle_name_or_path', type=str,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=0, help='chunk overlap (default: 0)')
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
    args = parser.parse_args()
    log.info(f'args: {args}')

    main(args)
