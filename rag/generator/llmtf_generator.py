from rusBeIR.rag.generator.base import BaseGenerator, default_prompt_maker
from llmtf.base import LLM
from rusBeIR.utils.type_hints import RetrieverResults, GeneratorResults, Corpus, Queries, QueryPromptMaker
from tqdm.autonotebook import tqdm

from typing import Dict, List

class LLMTFGenerator(BaseGenerator):
    def __init__(self, llmtf_model: LLM):
        self.llmtf_model = llmtf_model

    def _batch_items(self, items: List, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _query_to_llmtf_input(self, query: str):
        return [{'role': 'user', 'content': query}]

    def generate(
        self,
        retriever_results: RetrieverResults,
        corpus: Corpus,
        queries: Queries,
        query_prompt_maker: QueryPromptMaker = default_prompt_maker,
        **kwargs
    ) -> GeneratorResults:
        generated_results = {}
        query_ids = list(queries.keys())


        prompts = [
            query_prompt_maker(query_id, retriever_results, corpus, queries)
            for query_id in query_ids
        ]

        generated = self.llmtf_model.generate_batch([
            self._query_to_llmtf_input(prompt)
            for prompt in prompts
        ])
        
        for query_id, generated_result in zip(query_ids, generated[1]):
            generated_results[query_id] = generated_result
            
        return generated_results
