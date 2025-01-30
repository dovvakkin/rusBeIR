import typing as tp

import numpy as np
from scipy.spatial.distance import cosine
from bert_score import score as bert_score

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch

from rusBeIR.rag.scoring.metrics.base import BaseMetric
import rusBeIR.utils.type_hints as type_hints


class BertScoreMetric(BaseMetric):
    def __call__(
            self, 
            generated_responses: type_hints.MetricResponses,
            reference_responses: type_hints.MetricReferences,
            **kwargs
        )-> tp.Dict[str, float]:

        bert_score_precision = []
        bert_score_recall = []
        bert_score_f1 = []

        for gen, all_ref in zip(generated_responses, reference_responses):
            generations = [gen] * len(all_ref)
            P, R, F1 = bert_score(generations, all_ref, lang='ru')
            
            bert_score_precision.append(P.max().item())
            bert_score_recall.append(R.max().item())
            bert_score_f1.append(F1.max().item())

        return {
            'bert_score_precision': np.mean(bert_score_precision),
            'bert_score_recall': np.mean(bert_score_recall),
            'bert_score_f1': np.mean(bert_score_f1)
        }


class SemanticSimilarityMetric(BaseMetric):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def __call__(
            self, 
            generated_responses: type_hints.MetricResponses,
            reference_responses: type_hints.MetricReferences,
            **kwargs
        ) -> tp.Dict[str, float]:

        similarities = []

        for gen, all_ref in zip(generated_responses, reference_responses):

            gen_embedding = self.model.encode(gen)
            ref_embeddings = self.model.encode(all_ref)
        
            intermediate_similarities = []
            for ref_embedding in ref_embeddings:
                similarity = 1 - cosine(gen_embedding, ref_embedding)
                intermediate_similarities.append(similarity)

            similarities.append(np.max(intermediate_similarities))
            
        return {'semantic_similarity': np.mean(similarities)}
    

class CustomSemanticSimilairty(BaseMetric):
    def __init__(self, model_name: str = 'bert-base-uncased'):
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
            embeddings = outputs.last_hidden_state.mean(dim=1) # TODO: clarify if division by attention mask is required here
            
        return embeddings
        
    def __call__(
            self, 
            generated_responses: type_hints.MetricResponses,
            reference_responses: type_hints.MetricReferences,
            **kwargs
        ) -> tp.Dict[str, float]:
        scores = []
        
        for gen, all_ref in zip(generated_responses, reference_responses):
            gen_emb = self._get_embeddings(gen)

            intermediate_scores = []

            for ref in all_ref:
                ref_emb = self._get_embeddings(ref)
            
                score = torch.cosine_similarity(gen_emb, ref_emb).item()
                intermediate_scores.append(score)
            
            scores.append(np.max(intermediate_scores))
            
        return {f'bertscore_custom | {self.model_name}': np.mean(scores)}