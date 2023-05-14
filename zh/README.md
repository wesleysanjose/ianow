# 应用截屏
![image](https://user-images.githubusercontent.com/28772823/238087007-dfd166c2-ca13-4254-9b2e-3349784d6513.jpg)
#
![image](https://user-images.githubusercontent.com/28772823/238087003-6818390c-e367-43e3-9353-f1e52edb2016.jpg)

# 利用本地文档知识和大语言模型进行问答（和 Langchain 集成）

此应用允许您使用语言模型 (LLM) 执行文档搜索。它包括从目录加载txt文档，利用ChromaDB将它们转换为向量库，并使用本地 LLM 模型基于这些文档进行问答。

请确保您有GPU，因为目前还不支持 CPU 上运行。
针对 vicuna-13b-v1.1 模型进行了测试，效果还不错。模型默认会以 fp16，bf16 （通过特定参数）或 8bit（通过特定参数）加载。

## 基于gradio的聊天应用
- 从 UI 上传文档，可以基于该文档知识进行聊天
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme_chat.md

## 基于websocket 流问答
- 服务器预先加载文档库，提供 websocket 流接口，客户端可以进行问答
https://github.com/wesleysanjose/ianow/blob/main/doc_search_llm/servers/readme.md

## python 应用代码
### conda 环境
推荐使用 conda 环境, 在 python 3.9, pytorch 1.17， cuda 11.7 进行了测试，使用了 Nvidia RTX 3090 GPU
```bash
conda create -n ianow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -n ianow
conda install cudatoolkit-dev=11.7.0 -n jsllama4b -c conda-forge -n ianow
conda activate ianow
pip install -r requirements.txt
```

### 如何使用
你可以使用 Huggingface 加载本地或者网络的 LLM 模型，代码基于 vicuna-13b-v1.1 进行了测试。

```bash
git clone https://github.com/wesleysanjose/ianow
cd ianow
python -m doc_search_llm.llm_doc_query --modle_name_or_path=<model_name_or_path> --docs_root=<docs_root> --query=<query>
```

#### 参数:

- `--modle_name_or_path`：这是用于查询本地或者网络 LLM 模型名称或路径，是必需的参数。
- `--chunk_size`：可以把上传的文档进一步分块，块大小默认值为 1000。
- `--chunk_overlap`：块之间可以重叠，其默认值为100。
- `--docs_root`：这是文档的根目录，是必需的参数。
- `--global_kwargs`：这此参数用于过滤要加载的文档，默认值为`**/*.txt`。
- `--query`：这是你的查询，用于在加载的文档的上下文中对 LLM 进行查询，这是必需的参数。
- `--load_in_8bit`：使用此标志以 8 位加载模型，这是可选的。
- `--bf16`：如果GPU支持，使用此标志以 bf16 加载模型，否则将以 fp16 加载。这是可选的。
- `--top_n_docs_feed_llm`：为了避免向 LLM 提供过多的文档，我们使用向量库进行初筛，只向 LLM 提供前 N 个最匹配的文档。默认值为 4。
- `--trust_remote_code`：使用此标志信任远程代码。这是可选的。


### 工作原理

该脚本执行以下操作：

1. 从指定目录加载文档。
2. 将文档转换为向量库。
3. 如果提供了查询，它将：
   - 加载指定的 LLM 模型。
   - 创建 LLM 管道。
   - 加载 Langchain QA 链。
   - 搜索前 N 个最佳匹配的文档以减小查询范围。
   - 通过提供最佳匹配的文档来运行 LLM 查询。

## 感谢
受以下项目启发
- https://github.com/hwchase17/langchain
- https://github.com/oobabooga/text-generation-webui
- https://github.com/LianjiaTech/BELLE
