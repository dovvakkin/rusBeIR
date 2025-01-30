from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.type_hints import RetrieverResults, GeneratorResults, Corpus, Queries, QueryPromptMaker
from rusBeIR.rag.generator.base import BaseGenerator, default_prompt_maker
from tqdm.autonotebook import tqdm

class HFGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4,
        **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        ).to(device)
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

    def _batch_items(self, items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

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
        
        for batch_query_ids in tqdm(list(self._batch_items(query_ids, self.batch_size))):

            batch_prompts = [
                query_prompt_maker(query_id, retriever_results, corpus, queries)
                for query_id in batch_query_ids
            ]
            
            batch_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            batch_generated_texts = self.tokenizer.batch_decode(
                batch_outputs,
                skip_special_tokens=True
            )
            
            for query_id, generated_text in zip(batch_query_ids, batch_generated_texts):
                generated_results[query_id] = generated_text
            
        return generated_results
