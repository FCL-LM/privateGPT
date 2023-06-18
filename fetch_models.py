#!/usr/bin/env python3
import os
import boto3

model_name = os.environ.get('MODEL_NAME', "ggml-gpt4all-j-v1.3-groovy.bin")
access_key = os.environ.get('S3_ACCESS_KEY', 'adminadmin')
secret_key = os.environ.get('S3_SECRET_KEY', 'adminadmin')
endpoint_url = os.environ.get('S3_ENDPOINT', 'http://localhost:8333')

print(f"model_name {model_name}")
print(f"access_key {access_key}")
print(f"secret_key {secret_key}")

# Retrieve the list of existing buckets
s3 = boto3.client(use_ssl = True,\
                    service_name = 's3',\
                    aws_access_key_id = access_key,\
                    aws_secret_access_key = secret_key,\
                    endpoint_url = endpoint_url)

pathfile = os.path.join("models/", model_name)

#control if the file exists already in local
if os.path.exists(pathfile):
    sys.exit("model already fetched. If the model was updated remove the old model and restart the app")


try:
    s3.download_file('models', model_name, pathfile)
except:
    raise Exception(f"fetch model {model_name} failed.")
