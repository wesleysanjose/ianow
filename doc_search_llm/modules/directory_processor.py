from utils.simple_logger import Log
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

# docs_root = "/home/missa/dev/"
# kwargs = "**/*.md"


class DirectoryProcessor:
    def __init__(self, docs_root, kwargs="**/*.*"):
        log.debug(f'Initializing directory processor with {docs_root}')
        log.debug(f'Global kwargs: {kwargs}')

        if not Path(docs_root).is_dir():
            log.error(f'Invalid directory path: {docs_root}')
            raise ValueError('Invalid directory path')
        if not isinstance(kwargs, str):
            log.error(f'Invalid global keyword: {kwargs}')
            raise ValueError('Invalid global keyword')

        self.docs_root = docs_root
        self.kwargs = kwargs
        self.docs = []

    def load(self, chunk_size = 0, chunk_overlap = 0):
        log.debug(f'Loading documents from {self.docs_root}')
        log.debug(f'chunk_size: {chunk_size}')
        log.debug(f'chunk_overlap: {chunk_overlap}')

        try:
            loader = DirectoryLoader(self.docs_root, self.kwargs)
            self.docs = loader.load()
            log.info(f'You have loaded {len(self.docs)} document(s)')
        except Exception as e:
            log.error(f'Error loading documents: {e}')
            raise e
        #print the first 80 characters of the all documents
        for i, doc in enumerate(self.docs):
            log.info(f'The document {i} contains {len(doc.page_content)} characters')

        if chunk_size > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.docs = text_splitter.split_documents(self.docs)
            log.info(f'Your original documents have been splitted into {len(self.docs)} documents')
        return self.docs