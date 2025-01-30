from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.type_hints import RetrieverResults, GeneratorResults, Corpus, Queries, QueryPromptMaker
from rusBeIR.rag.generator.base import BaseGenerator, default_prompt_maker

class HFGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            **kwargs
        ).to(device)
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(
        self,
        retriever_results: RetrieverResults,
        corpus: Corpus,
        queries: Queries,
        query_prompt_maker: QueryPromptMaker = default_prompt_maker,
        **kwargs
    ) -> GeneratorResults:
        generated_results = {}
        
        for query_id in queries:
            prompt = query_prompt_maker(query_id, retriever_results, corpus, queries)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            generated_results[query_id] = generated_text
            
        return generated_results
