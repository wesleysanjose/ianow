from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
import torch

# load all courses from the docs directory with the .txt extension
def load_all_courses(docs_root):
  loader = DirectoryLoader(docs_root, glob = "**/*.md")
  #loader = DirectoryLoader(docs_root, glob = "**/*.md")
  docs = loader.load()
  return docs

docs = load_all_courses("/home/missa/dev/")
print (f'You have {len(docs)} document(s) in your data')
print (f'There are {len(docs[0].page_content)} characters in your document')

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

print (f'Now you have {len(split_docs)} documents')

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'chroma_storage'
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
vectorstore.persist()

# Load the vectorstore from disk
#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "how langchain can be used to integrate with llama.cpp?"
results = vectorstore.similarity_search(query)

print(f'matched docs: {len(results)}')
#for i, result in enumerate(results):
#  print(f'matched doc {i}: {result}')

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

query = "how to use stable vicuna?"
docs = vectorstore.similarity_search(query, 3, include_metadata=True)

print(len(docs))
print(docs[0])

chain.run(input_documents=docs, question=query)
