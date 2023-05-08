from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log(__name__)

# load all courses from the docs directory with the .md extension
def load_all_courses(docs_root):
  loader = DirectoryLoader(docs_root, glob = "**/*.md")
  docs = loader.load()
  return docs

docs = load_all_courses("/home/missa/dev/")
log.info(f'You have {len(docs)} document(s) in your data')
log.info(f'There are {len(docs[0].page_content)} characters in your document')

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

log.info(f'Now you have {len(split_docs)} documents')

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'chroma_storage'
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
vectorstore.persist()

# Load the vectorstore from disk
#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "how langchain can be used to integrate with llama.cpp?"
results = vectorstore.similarity_search(query)

log.info(f'matched docs: {len(results)}')
#for i, result in enumerate(results):
#  log.info(f'matched doc {i}: {result}')

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("/home/missa/dev/bloomz-7b1-mt")
model = AutoModelForCausalLM.from_pretrained("/home/missa/dev/bloomz-7b1-mt", device_map="auto", load_in_8bit=True)
#model.to(device)
#model.half()

# model_id = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
llm = HuggingFacePipeline(pipeline=pipe)

chain = load_qa_chain(llm, chain_type="stuff")

query = "what's langchain?"
docs = vectorstore.similarity_search(query, 3, include_metadata=True)

log.info(len(docs))
log.info(docs[0])

print(chain.run(input_documents=docs, question=query))
