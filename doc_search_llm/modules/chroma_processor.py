from langchain.vectorstores import Chroma

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log.get_logger(__name__)

class ChromaProcessor:
    def __init__(self, embeddings=None, persist_directory='chroma_storage'):
        log.debug(f'Initializing vectorstore processor with {embeddings}')
        log.debug(f'Persisting vectorstore to {persist_directory}')

        if persist_directory and not os.path.isdir(persist_directory):
            raise ValueError('Invalid directory path provided')
        
        self.embeddings = None
        self.vectorstore = None
        self.persist_directory = persist_directory

    def convert_from_docs(self, docs):
        log.debug(f'Converting {len(docs)} documents to vectorstore')
        try:
            self.vectorstore = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_directory)
        except FileNotFoundError as fnf_error:
            log.error(f'Error creating vectorstore: {fnf_error}')
            self.vectorstore = None
            raise fnf_error
        except Exception as e:
            log.error(f'Unexpected error occurred while creating vectorstore: {e}')
            self.vectorstore = None
            raise e

    def load_from_disk(self):
        log.debug(f'Loading vectorstore from {self.persist_directory}')
        try:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            log.info(f'Vectorstore loaded with {len(self.vectorstore)} vectors')
        except FileNotFoundError as fnf_error:
            log.error(f'Error loading vectorstore: {fnf_error}')
            self.vectorstore = None
            raise fnf_error
        except Exception as e:
            log.error(f'Unexpected error occurred while loading vectorstore: {e}')
            self.vectorstore = None
            raise e

    def save(self):
        log.debug(f'Saving vectorstore to {self.persist_directory}')
        if self.vectorstore:
            try:
                self.vectorstore.persist()
            except Exception as e:
                log.error(f'Unexpected error occurred while saving vectorstore: {e}')
                raise e
        else:
            log.error('No vectorstore to save. Please ensure the vectorstore is loaded or created before saving.')
            raise Exception('No vectorstore to save')
