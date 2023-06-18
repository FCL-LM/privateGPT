#!/usr/bin/env python3
import os, sys
import boto3
from tqdm import tqdm

model_name = os.environ.get('MODEL_NAME', "ggml-gpt4all-j-v1.3-groovy.bin")
access_key = os.environ.get('S3_ACCESS_KEY', 'adminadmin')
secret_key = os.environ.get('S3_SECRET_KEY', 'adminadmin')
endpoint_url = os.environ.get('S3_ENDPOINT', 'http://localhost:8333')

print(f"Trying to retrieve model {model_name}...")

# Retrieve the list of existing buckets
s3 = boto3.client(use_ssl = True,\
                    service_name = 's3',\
                    aws_access_key_id = access_key,\
                    aws_secret_access_key = secret_key,\
                    endpoint_url = endpoint_url)

pathfile = os.path.join("models", model_name)

# Control if the file exists already in local
if os.path.exists(pathfile):
    sys.exit("Model already fetched. If the model was updated remove the old model and restart the app")

try:
    model_size = s3.head_object(Bucket='models', Key=model_name)['ContentLength']

    with tqdm(total=model_size, unit_scale=True, desc=model_name) as progressbar:
        s3.download_file(Bucket = 'models',\
                            Key = model_name,\
                            Filename = pathfile,\
                            Callback = lambda transf: progressbar.update(transf))
except:
    raise Exception(f"fetch model {model_name} failed.")