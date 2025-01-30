import typing as tp

import numpy as np
from scipy.spatial.distance import cosine
from bert_score import score as bert_score

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch

from rusBeIR.rag.scoring.metrics.base import BaseMetric


class BertScoreMetric(BaseMetric):
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        P, R, F1 = bert_score(generated_responses, reference_responses, lang='en')
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }


class SemanticSimilarityMetric(BaseMetric):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        gen_embeddings = self.model.encode(generated_responses)
        ref_embeddings = self.model.encode(reference_responses)
        
        similarities = []
        for gen_emb, ref_emb in zip(gen_embeddings, ref_embeddings):
            similarity = 1 - cosine(gen_emb, ref_emb)
            similarities.append(similarity)
            
        return {'semantic_similarity': np.mean(similarities)}
    

class CustomSemanticSimilairty(BaseMetric):
    def init(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
    def _get_embeddings(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
        
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        scores = []
        
        for gen, ref in zip(generated_responses, reference_responses):
            gen_emb = self._get_embeddings(gen)
            ref_emb = self._get_embeddings(ref)
            
            score = torch.cosine_similarity(gen_emb, ref_emb).item()
            scores.append(score)
            
        return {f'bertscore_custom | {self.model_name}': np.mean(scores)}