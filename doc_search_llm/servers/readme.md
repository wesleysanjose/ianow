# Query/Answer with local documents and local LLM via WebSocket streaming

This script provides a WebSocket server that uses a Language Model (LLM) for document searching. It allows clients to send search queries over WebSocket, and the server will respond answer based on the documents provided in docs_root.

## How to Use:

To run the WebSocket server, execute the script and provide the necessary command-line arguments:

```bash
python -m  --model_name_or_path model_path --docs_root docs_directory --global_kwargs **/*.txt
```

After the server has started, you can connect to it over WebSocket from your client and send your search queries. The server will respond with the answers.

From macbook, you can use websocat for testing
```bash
websocat ws://<WS_SERVER>:5000/ws
```

## Command-line Arguments:

The script supports a number of command-line arguments:

- `--model_name_or_path`: The name or path of the LLM model to be used for the query.
- `--chunk_size`: The size of chunks in which to divide the documents (default is 1000).
- `--chunk_overlap`: The overlap between chunks (default is 100).
- `--docs_root`: The root directory of the documents to be loaded.
- `--global_kwargs`: Global arguments to be passed to the directory processor (default is "**/*.txt").
- `--load_in_8bit`: If set, load the model in 8 bits.
- `--bf16`: If set, use bf16.
- `--doc_count_for_qa`: The number of documents to consider for question-answering (default is 4).
- `--port`: The port number on which to run the WebSocket server (default is 5000).
- `--trust_remote_code`: If set, trust remote code.
