from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import ElasticVectorSearch
from langchain.llms import GPT4All, LlamaCpp
import os

class GPTModel:
    __embeddings: HuggingFaceEmbeddings
    __db: ElasticVectorSearch
    __retriever: ElasticVectorSearch.as_retriever

    __llm: None

    __memory: ConversationBufferMemory
    __chain: ConversationalRetrievalChain.from_llm

    __last_answer: str
    __last_source_documents: str


    def __init__(self) -> None:
        load_dotenv()
        self.__embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
        self.__persist_directory = os.environ.get('PERSIST_DIRECTORY')

        self.__model_type = os.environ.get('MODEL_TYPE')
        
        model_name = os.environ.get('MODEL_NAME')
        self.__model_path = os.path.join("models/", model_name)
        self.__model_n_ctx = os.environ.get('MODEL_N_CTX')
        self.__model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
        self.__target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

        self.__n_cores = os.environ.get("N_CORES")
        if self.__n_cores is None:
            self.__n_cores = len(os.sched_getaffinity(0))

        # activate/deactivate the streaming StdOut callback for LLMs
        self.__callbacks = [StreamingStdOutCallbackHandler()]

        self.init_model()
    

    # ----- PROPERTIES -----

    # ---- ENV VARIABLES ----
    @property
    def embeddings_model_name(self):
        return self.__embeddings_model_name
    
    @property
    def persist_directory(self):
        return self.__persist_directory
    
    @property
    def model_type(self):
        return self.__model_type
    
    @property
    def model_path(self):
        return self.__model_path
    
    @property
    def model_n_ctx(self):
        return self.__model_n_ctx

    @property
    def model_n_batch(self):
        return self.__model_n_batch
    
    @property
    def target_source_chunks(self):
        return self.__target_source_chunks
    
    @property
    def n_cores(self):
        return self.__n_cores
    
    
    # llm internal data-structures 
    @property
    def embeddings(self):
        return self.__embeddings
    
    @property
    def callbacks(self):
        return self.__callbacks
    
    @embeddings.setter
    def embeddings(self, embeds: HuggingFaceEmbeddings):
        self.__embeddings = embeds

    @property
    def database(self):
        return self.__db

    @database.setter
    def database(self, new_db: ElasticVectorSearch):
        self.__db = new_db

    @property
    def retriever(self):
        return self.__retriever
    
    @retriever.setter
    def retriever(self, retriever: ElasticVectorSearch.as_retriever):
        self.__retriever = retriever
    
    @property
    def language_model(self):
        return self.__llm
    
    @language_model.setter
    def language_model(self, new_llm):
        self.__llm = new_llm

    @property
    def memory_buffer(self):
        return self.__memory
    
    @memory_buffer.setter
    def memory_buffer(self, buffer: ConversationBufferMemory):
        self.__memory = buffer

    @property
    def retrivial_chain(self):
        return self.__chain

    @retrivial_chain.setter
    def retrivial_chain(self, new_chain):
        self.__chain = new_chain


    # ---- FINAL OUTPUT VARIABLES ----
    # inference
    @property
    def answer(self):
        return self.__last_answer
    
    @answer.setter
    def answer(self, inference: str):
        self.__last_answer = inference
    
    # source documents
    @property
    def source_documents(self):
        return self.__last_source_documents
    
    @source_documents.setter
    def source_documents(self, source_docs: str):
        self.__last_source_documents = source_docs
    
    # ----- METHODS -----
    # Prepare embeddings, database and retriever
    def load_embeddings_retriever(self):
        self.__embeddings = HuggingFaceEmbeddings(model_name = self.embeddings_model_name)
        self.__db = ElasticVectorSearch(elasticsearch_url='http://elastic:adminadmin@127.0.0.1:9200',\
                                        index_name="test_index",\
                                        embedding=self.__embeddings)
        self.__retriever = self.database.as_retriever(search_kwargs={"k": self.target_source_chunks})
    

    # Prepare the LLM
    def load_llm_model(self):
        if self.model_type == "LlamaCpp":
            self.language_model = LlamaCpp(model_path = self.model_path,\
                                            n_threads = self.n_cores,\
                                            n_ctx = self.model_n_ctx,\
                                            n_batch = self.model_n_batch,\
                                            callbacks = self.callbacks,\
                                            verbose = False)
        elif self.model_type == "GPT4All":
            self.language_model = GPT4All(model = self.model_path,\
                                            n_threads = self.n_cores,\
                                            n_ctx = self.model_n_ctx,\
                                            backend = 'gptj',\
                                            n_batch = self.model_n_batch,\
                                            callbacks = self.callbacks,\
                                            verbose = False)
        else:
            sys.exit(f"Model {self.model_type} not supported!")


    # Preparing the memory-buffered chain
    def load_memory_chain(self):
        self.memory_buffer = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')                     # buffer containing the entire conversation history
        self.retrivial_chain = ConversationalRetrievalChain.from_llm(llm = self.language_model, retriever = self.retriever,\
                                                                     memory=self.memory_buffer, return_source_documents=True)
        

    def init_model(self):
        self.load_embeddings_retriever()
        self.load_llm_model()
        self.load_memory_chain()


    # Query the LLM model with a string
    def query_model(self, query: str):
        result = self.retrivial_chain(query)
        self.answer, self.source_documents = result['answer'], result['source_documents']

        # Print the relevant sources used for the answer
        for document in self.source_documents:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

        return self.answer