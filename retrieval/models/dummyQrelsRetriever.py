from llmtf.base import Base
from rusBeIR.utils.type_hints import Corpus, Queries, RetrieverResults, Qrels
from rusBeIR.beir.retrieval.search.base import BaseSearch

class DummyQrelsRetriever(BaseSearch):
    def __init__(self, qrels: Qrels):
        """
        dummy retriever that return qrels for any .search call

        this retriever is meant to be used for evaludation rag generator models
        on best possible retriever results (qrels from beir format dataset)
        """
        self.qrels = qrels
        
    def search(self, 
               corpus: Corpus,
               queries: Queries,
               top_k: int,
               **kwargs) -> RetrieverResults:
        return self.qrels


