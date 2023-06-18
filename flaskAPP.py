#!/usr/bin/env python3
import time
import logging
from flask import Flask, request, jsonify
from privateGPT import GPTModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class FlaskGPT:
    def __init__(self, gpt: GPTModel) -> None:
        self.__gpt_model = gpt
        self.__app = Flask(__name__)

        @self.app.route('/', methods=['POST'])
        def __query():
            return self.query()

    @property
    def app(self):
        return self.__app
    
    @property
    def gpt_model(self):
        return self.__gpt_model

    def query(self):
        answer = {}
        api_query = request.json['message']

        # Get the answer from the chain
        start = time.time()
        answer['answer'] = self.gpt_model.query_model(api_query)
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(api_query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        
        return jsonify(answer)
    
    def run(self):
        self.__app.run()

flask = FlaskGPT(GPTModel())
flask.run()