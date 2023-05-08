import traceback
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log.get_logger(__name__)

class VectorstoreProcessor:
    def __init__(self, embeddings_object=None, persist_directory='chroma_storage'):
        log.debug(f'Initializing vectorstore processor with {embeddings}')
        log.debug(f'Persisting vectorstore to {persist_directory}')
        if embeddings_object is None:
            embeddings_object = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        else:
            self.embeddings_object = embeddings_object
        self.vectorstore = None
        self.persist_directory = persist_directory

    def convert_from_docs(self, docs):
        log.debug(f'Converting {len(docs)} documents to vectorstore')
        try:
            self.vectorstore = Chroma.from_documents(docs, self.embeddings_object, persist_directory=self.persist_directory)
        except Exception as e:
            log.error(f'Error creating vectorstore: {e}')
            traceback.print_exc()
            self.vectorstore = None
            raise e

    def load_from_disk(self):
        log.debug(f'Loading vectorstore from {self.persist_directory}')
        try:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings_object)
            log.info(f'Vectorstore loaded with {len(self.vectorstore)} vectors')
        except Exception as e:
            log.error(f'Error loading vectorstore: {e}')
            self.vectorstore = None

    def save(self):
        log.debug(f'Saving vectorstore to {self.persist_directory}')
        if self.vectorstore is not None:
            self.vectorstore.persist()

    def similarity_search(self, query):
        log.debug(f'Performing similarity search for {query}')
        if self.vectorstore is not None:
            return self.vectorstore.similarity_search(query)
        else:
            log.error(f'Vectorstore is null')
            return None
