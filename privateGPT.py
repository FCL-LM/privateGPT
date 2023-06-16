#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import sys
import csv

load_dotenv()

n_cores = os.environ.get("N_CORES")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
performance_data_time_file = os.environ.get("PERFORMANCE_DATA_TIME_FILE")
performance_data_sources_file = os.environ.get("PERFORMANCE_DATA_SOURCES_FILE")

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    if n_cores is None:
        n_cores = len(os.sched_getaffinity(0))
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_threads=n_cores, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_threads=n_cores, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        #Print time taken to answer to file for benchmarking
        print(f"{round(end - start, 2)}")
        print(answer)
        with open(performance_data_time_file, "a") as f:
            f.write(f"{n_cores},{model_type + model_path},{embeddings_model_name},{round(end - start, 2)}\n")
        # Print the relevant sources used for the answer in a csv file
        if not args.hide_source:
            with open(performance_data_sources_file, "a") as f:
                writer = csv.writer(f)
                for document in docs:
                    writer.writerow([document.metadata["source"]])
                # print("\n> Sources:")
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
