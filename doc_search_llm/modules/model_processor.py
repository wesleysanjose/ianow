from utils.simple_logger import Log
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import torch

from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

# model processor class


class ModelProcessor:

    @staticmethod
    def load_gpt4all(args):
        log.debug(f'Loading model from {args}')
        
        try:
            # Callbacks support token-wise streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            # Verbose is required to pass to the callback manager

            # Make sure the model path is correct for your system!
            model = GPT4All(
                model=args.model_name_or_path, callback_manager=callback_manager, verbose=True
            )
        except Exception as e:
            log.error(
                f'Failed to load the model from {args.model_name_or_path}')
            log.error(f'Exception: {e}')
            raise e
        return model

    @staticmethod
    def load_llamacpp(args):
        log.debug(f'Loading model from {args}')
        
        try:
            # Callbacks support token-wise streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            # Verbose is required to pass to the callback manager

            # Make sure the model path is correct for your system!
            model = LlamaCpp(
                model_path=args.model_name_or_path, callback_manager=callback_manager, verbose=True
            )
            embeddings = LlamaCppEmbeddings(model_path=args.model_name_or_path)
        except Exception as e:
            log.error(
                f'Failed to load the model from {args.model_name_or_path}')
            log.error(f'Exception: {e}')
            raise e
        return model, embeddings

    # load the model
    @staticmethod
    def load_model(args):
        log.debug(f'Loading model from {args}')
        if args is None or not hasattr(args, 'model_name_or_path') or args.model_name_or_path is None:
            log.error(f'Invalid arguments or no model name path provided')
            raise ValueError('Invalid arguments or no model name path provided')
        else:
            # load the model from the path
            try:
                if args.load_in_8bit:
                    log.info("Model Loading in 8 bits")
                    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                                 device_map="auto",
                                                                 quantization_config=BitsAndBytesConfig(
                                                                     load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
                                                                 trust_remote_code=args.trust_remote_code)
                else:
                    log.info("Default model loading in fp16 or bf16")
                    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                                 device_map="auto",
                                                                 torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                                                                 trust_remote_code=args.trust_remote_code)

                if type(model) is LlamaForCausalLM:
                    tokenizer = LlamaTokenizer.from_pretrained(
                        args.model_name_or_path, clean_up_tokenization_spaces=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model_name_or_path, clean_up_tokenization_spaces=True)

            except Exception as e:
                log.error(
                    f'Failed to load the model from {args.model_name_or_path}')
                log.error(f'Exception: {e}')
                raise e
            return model, tokenizer
