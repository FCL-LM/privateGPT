"""Loading logic for loading documents from an s3 directory."""
import os
import json
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

access_key = os.environ.get('S3_ACCESS_KEY', 'adminadmin')
secret_key = os.environ.get('S3_SECRET_KEY', 'adminadmin')
endpoint_url = os.environ.get('S3_ENDPOINT', 'http://localhost:8333')

class S3FileLoader(BaseLoader):
    """Loading logic for loading documents from s3."""

    def __init__(self, bucket: str, key: str):
        """Initialize with bucket and key name."""
        self.bucket = bucket
        self.key = key

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it with `pip install boto3`."
            )
        
        s3 = boto3.client(use_ssl = True,\
                            service_name = 's3',\
                            aws_access_key_id = access_key,\
                            aws_secret_access_key = secret_key,\
                            endpoint_url = endpoint_url)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(self.bucket, self.key, file_path)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()


class S3DirectoryLoader(BaseLoader):
    """Loading logic for loading documents from s3."""

    def __init__(self, bucket: str, prefix: str = ""):
        """Initialize with bucket and key name."""
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        
        # Read variables from environment and exit in case of empty string
        
        s3 = boto3.client(use_ssl = True,\
                            service_name = 's3',\
                            aws_access_key_id = access_key,\
                            aws_secret_access_key = secret_key,\
                            endpoint_url = endpoint_url)
        
        s3_objs = s3.list_objects_v2(Bucket=self.bucket)
        docs = []

        if "Contents" in s3_objs:
            for obj in s3_objs.get('Contents'):
                loader = S3FileLoader(self.bucket, obj.get('Key'))
                docs.extend(loader.load())
        else:
            print("The bucket is empty.")

        return docs

