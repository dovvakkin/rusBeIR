from abc import abstractmethod
import typing as tp

import spacy
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize

from rusBeIR.rag.scoring.metrics.base import BaseMetric


class BaseRagMetric(BaseMetric):

    @abstractmethod
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str], 
                 context_passages: tp.List[str]) -> tp.Dict[str, float]:
        raise NotImplementedError()


class RAGContextRelevanceMetric(BaseRagMetric):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
    def __call__(self, 
                 generated_responses: tp.List[str], 
                 context_passages: tp.List[str],
                 reference_responses: tp.Optional[tp.List[str]]=None
        ) -> tp.Dict[str, float]:
        scores = []
        
        for gen, context in zip(generated_responses, context_passages):
            gen_doc = self.nlp(gen)
            context_doc = self.nlp(context)
            
            # Вычисляем семантическое сходство между ответом и контекстом
            similarity = gen_doc.similarity(context_doc)
            scores.append(similarity)
            
        return {'context_relevance': np.mean(scores)}


class SourceAttributionMetric(BaseRagMetric):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
    def _extract_entities(self, text: str) -> tp.Set[str]:
        doc = self.nlp(text)
        return {ent.text.lower() for ent in doc.ents}
        
    def __call__(self, 
                 generated_responses: tp.List[str],
                 context_passages: tp.List[str],
                 reference_responses: tp.Optional[tp.List[str]]=None
        ) -> tp.Dict[str, float]:
        scores = []
        
        for gen, context in zip(generated_responses, context_passages):
            gen_entities = self._extract_entities(gen)
            context_entities = self._extract_entities(context)
            
            if not gen_entities:
                scores.append(1.0)  # Если нет сущностей, считаем ответ корректным
                continue
                
            # Вычисляем процент сущностей из ответа, которые есть в контексте
            attribution_score = len(gen_entities & context_entities) / len(gen_entities)
            scores.append(attribution_score)
            
        return {'source_attribution': np.mean(scores)}


class FactualConsistencyMetric(BaseRagMetric):
    def __init__(self):
        self.qa_pipeline = pipeline('question-answering')
        self.nlp = spacy.load('en_core_web_lg')
        
    def _extract_facts(self, text: str) -> tp.List[str]:
        """Извлекает факты из текста в виде простых предложений"""
        doc = self.nlp(text)
        facts = []
        
        for sent in doc.sents:
            # Простой эвристический подход: предложения с подлежащим и сказуемым
            if any(token.dep_ == 'nsubj' for token in sent) and \
               any(token.pos_ == 'VERB' for token in sent):
                facts.append(sent.text)
                
        return facts
        
    def __call__(self, 
                 generated_responses: tp.List[str], 
                 context_passages: tp.List[str], 
                 reference_responses: tp.Optional[tp.List[str]]=None
        ) -> tp.Dict[str, float]:
        consistency_scores = []
        
        for gen, context in zip(generated_responses, context_passages):
            facts = self._extract_facts(gen)
            if not facts:
                consistency_scores.append(1.0)
                continue
                
            fact_scores = []
            for fact in facts:
                # Пробуем найти подтверждение каждого факта в контексте
                try:
                    result = self.qa_pipeline(
                        question=fact,
                        context=context
                    )
                    fact_scores.append(result['score'])
                except:
                    fact_scores.append(0.0)
                    
            consistency_scores.append(np.mean(fact_scores))
            
        return {'factual_consistency': np.mean(consistency_scores)}


class HallucinationDetectionMetric(BaseRagMetric):
    def __init__(self):
        self.nli_pipeline = pipeline('zero-shot-classification')
        self.nlp = spacy.load('en_core_web_lg')
        
    def __call__(self, 
                 generated_responses: tp.List[str],
                 context_passages: tp.List[str],
                 reference_responses: tp.Optional[tp.List[str]]=None
        ) -> tp.Dict[str, float]:
        hallucination_scores = []
        
        for gen, context in zip(generated_responses, context_passages):
            # Разбиваем ответ на предложения
            sentences = sent_tokenize(gen)
            
            sent_scores = []
            for sent in sentences:
                # Проверяем, следует ли предложение из контекста
                result = self.nli_pipeline(
                    sent,
                    candidate_labels=['entailment', 'contradiction'],
                    premise=context
                )
                
                # Вычисляем score как вероятность entailment
                score = result['scores'][result['labels'].index('entailment')]
                sent_scores.append(score)
                
            hallucination_scores.append(np.mean(sent_scores))
            
        return {'hallucination_score': np.mean(hallucination_scores)}
    

class AnswerRelevanceMetric(BaseMetric):
    def __init__(self):
        self.nli_pipeline = pipeline('zero-shot-classification')
        
    def __call__(self, 
                 generated_responses: tp.List[str], 
                 questions: tp.List[str], 
                 context_passages: tp.List[str],
                 reference_responses: tp.Optional[tp.List[str]]=None
        ) -> tp.Dict[str, float]:
        relevance_scores = []
        
        for gen, question, context in zip(generated_responses, questions, context_passages):
            # Проверяем, отвечает ли генерация на вопрос
            answer_relevance = self.nli_pipeline(
                gen,
                candidate_labels=['relevant', 'irrelevant'],
                premise=question
            )
            
            relevance_score = answer_relevance['scores'][
                answer_relevance['labels'].index('relevant')
            ]
            
            relevance_scores.append(relevance_score)
            
        return {'answer_relevance': np.mean(relevance_scores)}
