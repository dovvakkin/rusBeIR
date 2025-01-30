import typing as tp
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from rouge_score import rouge_scorer # pip install rouge-score
from comet import download_model, load_from_checkpoint  # pip install unbabel-comet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # pip install nltk
from nltk.translate.meteor_score import meteor_score

from rusBeIR.rag.scoring.metrics.base import BaseMetric

class RougeMetric(BaseMetric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        scores = {
            'rouge1_precision': 0.0,
            'rouge1_recall': 0.0,
            'rouge1_f1': 0.0,
            'rouge2_precision': 0.0,
            'rouge2_recall': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_precision': 0.0,
            'rougeL_recall': 0.0,
            'rougeL_f1': 0.0,
        }
        
        for gen, ref in zip(generated_responses, reference_responses):
            results = self.scorer.score(gen, ref)
            
            scores['rouge1_precision'] += results['rouge1'].precision
            scores['rouge1_recall'] += results['rouge1'].recall
            scores['rouge1_f1'] += results['rouge1'].fmeasure
            
            scores['rouge2_precision'] += results['rouge2'].precision
            scores['rouge2_recall'] += results['rouge2'].recall
            scores['rouge2_f1'] += results['rouge2'].fmeasure
            
            scores['rougeL_precision'] += results['rougeL'].precision
            scores['rougeL_recall'] += results['rougeL'].recall
            scores['rougeL_f1'] += results['rougeL'].fmeasure
            
        n = len(generated_responses)
        return {k: v/n for k, v in scores.items()}


class BleuMetric(BaseMetric):
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        smooth = SmoothingFunction()
        scores = []
        
        for gen, ref in zip(generated_responses, reference_responses):
            gen_tokens = gen.split()
            ref_tokens = [ref.split()]
            score = sentence_bleu(ref_tokens, gen_tokens, 
                                smoothing_function=smooth.method1)
            scores.append(score)
            
        return {'bleu': np.mean(scores)}


class MeteorMetric(BaseMetric):
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        scores = []
        
        for gen, ref in zip(generated_responses, reference_responses):
            score = meteor_score([ref.split()], gen.split())
            scores.append(score)
            
        return {'meteor': np.mean(scores)}


class CIDErMetric(BaseMetric):
    def __init__(self, n=4):
        self.n = n
        self.stemmer = PorterStemmer()
        
    def _compute_ngrams(self, text: str, n: int) -> Counter:
        tokens = word_tokenize(text.lower())
        stems = [self.stemmer.stem(token) for token in tokens]
        ngrams = zip(*[stems[i:] for i in range(n)])
        return Counter(ngrams)
        
    def _compute_tf_idf(self, generated: tp.List[str], reference: tp.List[str]):
        doc_freq = Counter()
        term_freq = []
        
        # Compute document frequency
        for ref in reference:
            ngrams = self._compute_ngrams(ref, self.n)
            doc_freq.update(ngrams.keys())
            term_freq.append(ngrams)
            
        # Compute IDF
        num_docs = len(reference)
        idf = {ngram: np.log((num_docs + 1) / (df + 1)) for ngram, df in doc_freq.items()}
        
        return term_freq, idf
        
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        scores = []
        
        for gen, ref in zip(generated_responses, reference_responses):
            term_freq, idf = self._compute_tf_idf([gen], [ref])
            
            gen_ngrams = self._compute_ngrams(gen, self.n)
            ref_ngrams = self._compute_ngrams(ref, self.n)
            
            # Compute vectors
            gen_vec = np.array([gen_ngrams[ng] * idf.get(ng, 0) for ng in gen_ngrams])
            ref_vec = np.array([ref_ngrams[ng] * idf.get(ng, 0) for ng in ref_ngrams])
            
            # Compute cosine similarity
            norm_gen = np.linalg.norm(gen_vec)
            norm_ref = np.linalg.norm(ref_vec)
            
            if norm_gen == 0 or norm_ref == 0:
                score = 0
            else:
                score = np.dot(gen_vec, ref_vec) / (norm_gen * norm_ref)
            
            scores.append(score)
            
        return {'cider': np.mean(scores)}
    

class COMETMetric(BaseMetric):
    def __init__(self):
        model_path = download_model('wmt20-comet-da')
        self.model = load_from_checkpoint(model_path)
        
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        data = [{
            'src': '',  # In our case, we don't have source text
            'mt': gen,
            'ref': ref
        } for gen, ref in zip(generated_responses, reference_responses)]
        
        scores = self.model.predict(data, batch_size=8, gpus=1)
        return {'comet': np.mean(scores)}
