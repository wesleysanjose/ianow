# Document Search with LLM

This script allows you to perform document search using a Language Model (LLM). It involves loading documents from a directory, converting them into a vectorstore, and using an LLM model to answer queries based on these documents.

## How to use

You can use the script by running the following command:

```bash
python <script_name.py> --modle_name_or_path=<model_name_or_path> --docs_root=<docs_root> --query=<query>
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

### Example:

```bash
python <script_name.py> --modle_name_or_path="gpt-2" --docs_root="/home/user/documents" --query="What is AI?"
```

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

## Error Handling

The script has robust error handling and logging, which will trace any issues that might occur during its execution. If any error occurs during loading the documents, converting documents to vectorstore, or loading the model, it will be logged and the script will stop execution.
