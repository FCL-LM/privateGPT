#!/usr/bin/env python3
import os, sys
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

# Load the s3 bucket containing the documents to ingest
s3_source_loader = s3_override.S3DirectoryLoader("sources")

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

# Retrieves only the new documents that are not already in VectorStore
def filter_documents(vector_docs: List[str]) -> List[Document]:
    print(f"Loading documents from bucket s3://{s3_source_loader.bucket}")
    # Take documents from s3 bucket
    documents = s3_source_loader.load()

    if not documents:
        print("No new documents to load")
        sys.exit(0)

    # Control which docs have been already processed 
    new_docs = []
    for doc in documents:
        doc_source = os.path.basename(doc.dict()['metadata']['source'])
        if doc_source not in vector_docs:
            new_docs.append(doc)

    return new_docs


# Load new documents and split in chunks
# if vector_docs is empty, then documents are loaded from the s3 bucket
def process_documents(vector_docs: List[str] = []) -> List[Document]:
    new_documents = filter_documents(vector_docs)
    print(f"Loaded {len(new_documents)} new documents from bucket s3://{s3_source_loader.bucket}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(new_documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    
    return texts


# Checks if vectorstore exists
def does_vectorstore_exist() -> bool:
    es = Elasticsearch(elastic_endpoint)

    if es.indices.exists(index=elastic_index):
        res = es.cat.count(index=elastic_index)
        n_docs = int(res[2])

        # At least 3 docs are needed in a working vectorstore
        if n_docs > 3:
            return True

    return False


# Retrievies list of the documents in VectorStore
def get_vector_documents() -> set:
    es = Elasticsearch(elastic_endpoint)

    es_response = elastic_scan(
        es,
        index = elastic_index,
        query = {"query": { "match_all" : {} }}
    )

    # List of all documents already in VectorStore
    vector_files = []

    for record in es_response:
        record_source = record['_source']['metadata']['source']
        file_bn = os.path.basename(record_source)

        vector_files.append(file_bn)

    return vector_files


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name = embeddings_model_name)

    if does_vectorstore_exist():
        # Update and store vectorstore on ElasticSearch DB
        print(f"Appending to existing vectorstore at {elastic_index}")
        db =  ElasticVectorSearch(elasticsearch_url = elastic_endpoint,\
                                    index_name = elastic_index,\
                                    embedding = embeddings)
        
        vector_docs = get_vector_documents()
        texts = process_documents(vector_docs)
        if len(texts) > 0:
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
    else:
        # Create and store VectorStore on ElasticSearch DB
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = ElasticVectorSearch.from_documents(texts, embeddings,\
                                                elasticsearch_url = elastic_endpoint,\
                                                index_name = elastic_index)

    print(f"Ingestion complete! You can now run privateGPT to query your documents")


if __name__ == "__main__":
    main()
