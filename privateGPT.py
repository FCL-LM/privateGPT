#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time
import logging
from flask import Flask, request, jsonify

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
load_dotenv()

global n_cores

from constants import CHROMA_SETTINGS

app = Flask(__name__)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()]
n_cores = os.environ.get("N_CORES")
if n_cores is None:
    n_cores = len(os.sched_getaffinity(0))

# Prepare the LLM
if model_type == "LlamaCpp":
    llm = LlamaCpp(model_path=model_path, n_threads=n_cores, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
elif model_type == "GPT4All":
    llm = GPT4All(model=model_path, n_threads=n_cores, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
else:
    print(f"Model {model_type} not supported!")
    exit

# Preparing the memory-byffered chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')                     # buffer containing the entire conversation history
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True)

# Questions and answers via REST apis
@app.route('/', methods=['POST'])
def query():
    answer = {}
    api_query = request.json['message']

    # Get the answer from the chain
    start = time.time()
    res = qa(api_query)
    answer['answer'], docs = res['answer'], res['source_documents']
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(api_query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)

    return jsonify(answer)

app.run()