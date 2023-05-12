# Query/Answer Bot using local Documents and local LLM model through Gradio

This script provides an interactive interface using Gradio for query/answer by using local document as knowledge base with a Language Model (LLM). Users can upload a document file and chat with the bot, which uses the LLM to answer based on the knowledge of the documents.

## How to Use:

1. Run the script with the necessary command-line arguments.

2. The Gradio interface will launch in your default web browser.

3. Upload a document file using the file upload feature.

4. Chat with the bot using the chat interface.

## Command-line Arguments:

The script supports a number of command-line arguments:

- `--model_name_or_path`: The name or path of the LLM model to be used for the query.
- `--chunk_size`: The size of chunks in which to divide the documents (default is 1000).
- `--chunk_overlap`: The overlap between chunks (default is 100).
- `--load_in_8bit`: If set, load the model in 8 bits.
- `--bf16`: If set, use bf16.
- `--top_n_docs_feed_llm`: To avoid feeding too many documents to LLM, only top N best-matched documents are fed (default is 4).
- `--port`: The port number on which to run the Gradio interface (default is 7860).
- `--server_name`: The server name on which to run the Gradio interface (default is '0.0.0.0').
- `--trust_remote_code`: If set, trust remote code.

## Running the Gradio Interface:

To run the Gradio interface, execute the script and provide the necessary command-line arguments:

```bash
python -m doc_search_llm.servers.gr_server --model_name_or_path model_path
```

After the Gradio interface has launched, you can interact with it in your web browser (http://<CHAT_SERVER>:7860):

1. Upload a document file.
![image](https://user-images.githubusercontent.com/28772823/238087007-dfd166c2-ca13-4254-9b2e-3349784d6513.jpg)
2. Use the chat interface to send query to the bot. The bot will respond with the answer based on the knowledge from the uploaded documents.
![image](https://user-images.githubusercontent.com/28772823/238087003-6818390c-e367-43e3-9353-f1e52edb2016.jpg)
