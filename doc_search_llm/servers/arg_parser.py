from utils.simple_logger import Log

import argparse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

def load_parser():
    # process command line arguments
    parser = argparse.ArgumentParser(description='Process the arguments.')
    parser.add_argument('--modle_name_or_path', type=str, required=True,
                        help='model name or path for LLM query')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int,
                        default=100, help='chunk overlap (default: 100)')
    parser.add_argument('--docs_root', type=str,help='docs root directory')
    parser.add_argument('--global_kwargs', type=str,
                        default="**/*.txt", help='global kwargs (default: **/*.txt')
    parser.add_argument('--query', type=str,
                        help='query string, used to query against the docs')
    parser.add_argument(
        '--load_in_8bit', action='store_true', help='Load in 8 bits')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--doc_count_for_qa', type=int,
                        default=4,  help='doc count for QA')
    parser.add_argument('--port', type=int, default=5000, help='port number')
    parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')

    return parser.parse_args()
