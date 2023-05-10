import traceback
from utils.simple_logger import Log
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import torch


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

# model processor class


class ModelProcessor:

    @staticmethod
    def load_model(args):
        log.debug(f'Loading model from {args}')
        if args is not None and len(args) > 0 and args.modle_name_or_path is not None:
            # load the model from the path
            try:
                if args.load_in_8bit:
                    log.info("Model Loading in 8 bits")
                    model = AutoModelForCausalLM.from_pretrained(args.modle_name_or_path,
                                                                 device_map="auto",
                                                                 quantization_config=BitsAndBytesConfig(
                                                                     load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
                                                                 trust_remote_code=True)
                else:
                    log.info("Default model loading in fp16 or bf16")
                    model = AutoModelForCausalLM.from_pretrained(args.modle_name_or_path,
                                                                 device_map="auto",
                                                                 torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                                                                 trust_remote_code=True)

                if type(model) is LlamaForCausalLM:
                    tokenizer = LlamaTokenizer.from_pretrained(
                        args.modle_name_or_path, clean_up_tokenization_spaces=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.modle_name_or_path, clean_up_tokenization_spaces=True)

            except Exception as e:
                log.error(
                    f'Failed to load the model from {args.modle_name_or_path}')
                log.error(f'Exception: {e}')
                log.error(traceback.format_exc())
                sys.exit(1)
        else:
            log.error(f'No model name path provided')
            sys.exit(1)
        return model, tokenizer
