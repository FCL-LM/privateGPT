#!/usr/bin/env python3
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from privateGPT import GPTModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class FlaskGPT:
    def __init__(self, gpt: GPTModel) -> None:
        self.__gpt_model = gpt
        self.__app = Flask(__name__)
        CORS(self.__app)

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
        api_query = request.get_json()
        # api_message = api_query['messages'][0]['content']['parts'][0]
        print(api_query)

        answer = self.gpt_model.query_model(api_query['message'])

        response = jsonify(answer)
        # api_query = request.get_json()
        # print(api_query)
        # api_message = api_query['messages'][0]['content']['parts'][0]

        # # Get the answer from the chain
        # start = time.time()
        # answer['data']['text'] = self.gpt_model.query_model(api_message)
        # answer['type'] = 'success'
        # end = time.time()

        # # Print the result
        # print("\n\n> Question:")
        # print(api_message)
        # print(f"\n> Answer (took {round(end - start, 2)} s.):")
        
        return response
    
    def run(self):
        self.__app.run()

flask = FlaskGPT(GPTModel())
flask.run()