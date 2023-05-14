# Screenshots
![image](https://user-images.githubusercontent.com/28772823/238087007-dfd166c2-ca13-4254-9b2e-3349784d6513.jpg)
#
![image](https://user-images.githubusercontent.com/28772823/238087003-6818390c-e367-43e3-9353-f1e52edb2016.jpg)

## [Chinese 中文](./zh/README.md)

<details>
<summary> Document Chat using Command Line</summary>
# Query/Answer using local documents and local LLM (Langchain integration)

This script allows you to perform document search using a Language Model (LLM). It involves loading documents from a directory, converting them into a vectorstore, and using an LLM model to answer queries (as arg passed through the application) based on these documents.

Please make sure you have GPU available because the code is not ready for CPU.
The test was done against vicuna-13b-v1.1 model and result looks good. The model by default is loaded as fp16/bf16 if specified, or even 8bits through the specificied argument.

## gradio chat code
- upload a document from UI and bot will answer the query based on the knowledge in the document
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme_chat.md

## websocket streaming chat code
- server pre-loads documents from specified directory, can answer the query from client through streaming
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme.md

## python appliction code
### conda env
conda env is recommended, the code is tested against python 3.9, pytorch 1.17 and cuda 11.7 on a Nvidia RTX 3090.
```bash
conda create -n ianow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -n ianow
conda install cudatoolkit-dev=11.7.0 -n jsllama4b -c conda-forge -n ianow
conda activate ianow
pip install -r requirements.txt
```

### How to use

You can use the LLM model in local by providing a sys path or download the model from HF by using model name. The code is tested with vicuna-13b-v1.1.

```bash
git clone https://github.com/wesleysanjose/ianow
cd ianow
python -m doc_search_llm.llm_doc_query --modle_name_or_path=<model_name_or_path> --docs_root=<docs_root> --query=<query>
```

#### Parameters:

- `--modle_name_or_path`: This is the model name or path for the LLM query. This is a required argument.
- `--chunk_size`: This is the chunk size and the default value is 1000.
- `--chunk_overlap`: This is the chunk overlap and the default value is 100.
- `--docs_root`: This is the root directory of your documents. This is a required argument.
- `--global_kwargs`: This is the global keyword arguments and the default value is `**/*.txt`. This is used to filter the documents to be loaded.
- `--query`: This is your query string, used to query against LLM with the context of loaded documents. This is a required argument.
- `--load_in_8bit`: Use this flag to load the model in 8 bit. It is optional.
- `--bf16`: Use this flag to load the model in bf16 if the device supports it, otherwise it will load in fp16. It is optional.
- `--top_n_docs_feed_llm`: To avoid feeding the LLM too many documents, we only feed the top N best matched documents to LLM. The default value is 4.
- `--trust_remote_code`: Use this flag to trust remote code. It is optional.

### How it works

The script does the following:

1. Loads the documents from the specified directory.
2. Converts the documents to a vectorstore.
3. If a query is provided, it:
   - Loads the specified LLM model.
   - Creates the LLM pipeline.
   - Loads the QA chain.
   - Searches for the top N best matched documents to reduce the scope.
   - Runs the LLM query by feeding the best matched documents.

## Thanks
inspired by
- https://github.com/hwchase17/langchain
- https://github.com/oobabooga/text-generation-webui
- https://github.com/LianjiaTech/BELLE
</details>

<details>
<summary>Document Chant using Websocket Streaming</summary>
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

</details>

<details>
<Summary>Document chat using Gradio UI</Summary>

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

</details>