# Query/Answer using local documents and local LLM (Langchain integration)

This script allows you to perform document search using a Language Model (LLM). It involves loading documents from a directory, converting them into a vectorstore, and using an LLM model to answer queries (as arg passed through the application) based on these documents.

Please make sure you have GPU available because the code is not ready for CPU.
The test was done against viccuna-13b-v1.1 model and result looks good. The model by default is loaded as fp16/bf16 if specified, or even 8bits through the specificied argument.

## gradio chat code
- upload a document from UI and bot will answer the query based on the knowledge in the document
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme_chat.md

## websocket streaming chat code
- server pre-loads documents from specified directory, can answer the query from client through streaming
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme.md

## conda env
conda env is recommended, the code is tested against python 3.9, pytorch 1.17 and cuda 11.7 on a Nvidia RTX 3090.
```bash
conda create -n ianow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -n ianow
conda install cudatoolkit-dev=11.7.0 -n jsllama4b -c conda-forge -n ianow
conda activate ianow
pip install -r requirements.txt
```

## How to use

You can use the LLM model in local by providing a sys path or download the model from HF by using model name. The code is tested with viccuna-13b-v1.1.

```bash
git clone https://github.com/wesleysanjose/ianow
cd ianow
python -m doc_search_llm.llm_doc_query --modle_name_or_path=<model_name_or_path> --docs_root=<docs_root> --query=<query>
```

### Parameters:

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

## How it works

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
