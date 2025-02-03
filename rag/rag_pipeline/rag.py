from rusBeIR.beir.retrieval.search.base import BaseSearch
from rusBeIR.utils.type_hints import Corpus, Queries
from rusBeIR.rag.generator.base import BaseGenerator
from typing import Dict

class RAG:
    def __init__(self, retriever: BaseSearch, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def retrieve_and_generate(
        self,
        corpus: Corpus,
        queries: Queries,
        retriever_kwargs_dict: Dict = {},
        generator_kwargs_dict: Dict = {}
    ):  # seem to be bug in python cause -> Tuple[RetrieverResults, GeneratorResults] cause error
        retriever_results = self.retriever.search(
            corpus=corpus,
            queries=queries,
            **retriever_kwargs_dict
        )
        generator_results = self.generator.generate(
            retriever_results=retriever_results,
            corpus=corpus,
            queries=queries,
            **generator_kwargs_dict
        )

        return retriever_results, generator_results
