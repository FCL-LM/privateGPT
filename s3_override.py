"""Loading logic for loading documents from an s3 directory."""
import os
import json
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

credentials_file = os.environ.get('CREDENTIALS_FILE', 'credentials.json')
endpoint_url = os.environ.get('ENDPOINT_URL', 'http://localhost:9000')

s3_key = json.load(open(credentials_file))

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
        
        if s3_key:
            s3 = boto3.client(use_ssl = True,\
                                service_name = 's3',\
                                aws_access_key_id = s3_key.get('accessKey'),\
                                aws_secret_access_key = s3_key.get('secretKey'),\
                                endpoint_url = endpoint_url)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = f"{temp_dir}/{self.key}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                s3.download_file(self.bucket, self.key, file_path)
                loader = UnstructuredFileLoader(file_path)
                return loader.load()
        else:
            raise Exception('No credential file has been found,\
                            cannot load file from S3 bucket')

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
        
        if s3_key:
            s3 = boto3.client(use_ssl = True,\
                                service_name = 's3',\
                                aws_access_key_id = s3_key.get('accessKey'),\
                                aws_secret_access_key = s3_key.get('secretKey'),\
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
        else:
            raise Exception('No credential file has been found,\
                            cannot load directory from S3 bucket')
