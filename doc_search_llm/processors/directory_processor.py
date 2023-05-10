from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.simple_logger import Log
log = Log.get_logger(__name__)

# docs_root = "/home/missa/dev/"
# global_kwargs = "**/*.md"
class DirectoryProcessor:
    def __init__(self, docs_root, global_kwargs = "**/*.*"):
        log.debug(f'Initializing directory processor with {docs_root}')
        log.debug(f'Global kwargs: {global_kwargs}')

        self.docs_root = docs_root
        self.global_kwargs = global_kwargs
        self.docs = []

    def load(self, chunk_size = 0, chunk_overlap = 0):
        log.debug(f'Loading documents from {self.docs_root}')
        log.debug(f'chunk_size: {chunk_size}')
        log.debug(f'chunk_overlap: {chunk_overlap}')

        loader = DirectoryLoader(self.docs_root, self.global_kwargs)
        self.docs = loader.load()
        log.info(f'You have loaded {len(self.docs)} document(s)')
        #print the first 80 characters of the all documents
        for i, doc in enumerate(self.docs):
            log.debug(f'The first 80 characters of the document {i}: {doc.page_content[:80]}')
            log.debug(f'The document {i} contains {len(doc.page_content)} characters')

        if chunk_size > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.docs = text_splitter.split_documents(self.docs)
            log.info(f'Your original documents have been splitted into {len(self.docs)} documents')
        return self.docs