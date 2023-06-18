#!/usr/bin/env python3
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan as elastic_scan
import s3_override 
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

load_dotenv()

# Load environment variables
elastic_endpoint = os.environ.get('ELASTIC_ENDPOINT')
elastic_index = os.environ.get('ELASTIC_INDEX')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50

s3_source_directory = s3_override.S3DirectoryLoader("sources")

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(ignored_files: List[str]) -> List[Document]:
    print(f"Loading documents from bucket s3://{s3_source_directory.bucket}")
    documents = s3_source_directory.load()
    if not documents:
        print("No new documents to load")
        sys.exit(0)
    
    filtered_files = [f for f in documents if os.path.basename(f.dict()['metadata']['source']) not in ignored_files]
    return filtered_files


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    documents = load_documents(ignored_files)
    print(f"Loaded {len(documents)} new documents from bucket s3://{s3_source_directory.bucket}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist() -> bool:
    """
    Checks if vectorstore exists
    """
    es = Elasticsearch(elastic_endpoint)

    if es.indices.exists(index=elastic_index):
        res = es.cat.count(index=elastic_index)

        if int(res[2]) > 3:
            return True

    return False

def get_sources_metadata() -> set:
    es = Elasticsearch(elastic_endpoint)

    es_response = elastic_scan(
        es,
        index=elastic_index,
        query={"query": { "match_all" : {}}},
    )

    sources = set()

    for d in es_response:
        sources.add(os.path.basename(d['_source']['metadata']['source']))

    return sources

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist():
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {elastic_index}")
        db =  ElasticVectorSearch(elasticsearch_url=elastic_endpoint,\
                                        index_name=elastic_index,\
                                        embedding=embeddings)
        
        sources = get_sources_metadata()

        texts = process_documents(list(sources))

        if len(texts) > 0:
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = ElasticVectorSearch.from_documents(texts, embeddings, elasticsearch_url=elastic_endpoint, index_name=elastic_index)

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
