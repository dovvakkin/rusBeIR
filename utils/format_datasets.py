import typing as tp
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

import rusBeIR.utils.type_hints as type_hints


class BeirDataset:

    def __init__(
        self,
        corpus: type_hints.Corpus,
        queries: type_hints.Queries,
        qrels: type_hints.Qrels,
        ground_truth: type_hints.GroundTruth,
    ):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.ground_truth = ground_truth

    

    def _convert_custom_to_beir(df: pd.DataFrame) -> tp.Tuple[tp.Dict, tp.Dict, tp.Dict]:
        """
        Converts dataframe to BEIR format
        
        Returns:
            corpus: Dict[str, Dict[str, str]] - documents {doc_id: {'text': text, 'title': ''}}
            queries: Dict[str, str] - queries {query_id: query_text}
            qrels: Dict[str, Dict[str, int]] - relevance {query_id: {doc_id: score}}
        """
        # Create index of uniq documents
        document_index = {}  # text -> doc_id
        corpus = {}
        current_doc_id = 0
        
        for _, row in df.iterrows():
            for context in row['context']:
                doc_text = context['chunk']
                if doc_text not in document_index:
                    doc_id = str(current_doc_id)
                    document_index[doc_text] = doc_id
                    corpus[doc_id] = {
                        'text': doc_text,
                        'title': ''  # no titles in this dataset
                    }
                    current_doc_id += 1
        
        # Format queries & qrels
        queries = {}
        qrels = defaultdict(dict)
        
        for _, row in df.iterrows():
            query_id = str(row['id'])
            
            queries[query_id] = row['question']
            
            for context in row['context']:
                if context['is_relevant']:
                    doc_text = context['chunk']
                    doc_id = document_index[doc_text]
                    qrels[query_id][doc_id] = 1  # True & False relevance
        
        qrels = dict(qrels)
        
        return corpus, queries, qrels
    
    def _get_all_possible_answers(row) -> tp.Set[str]:
        """
            Collects all possible answers
        """
        answers = set()
        if isinstance(row['answers'], list):
            answers.update(row['answers'])
        if isinstance(row['normalized_answers'], list):
            answers.update(row['normalized_answers'])
        return answers
    
    def _create_beir_dataset_with_answers(df: pd.DataFrame) -> tp.Tuple[tp.Dict, tp.Dict, tp.Dict, tp.Dict]:
        """
            Creates BEIR with extra answers set
            
            Returns:
                corpus, queries, qrels, answers, where

                corpus: Dict[str, Dict[str, str]] - {document_id: {text: document's text, title: document's title}}
                queries: Dict[str, str] - {query_id: text of query}
                qrels: Dict[str, Dict[str, int]] - {query_id: {document_id: relevance}}
                answers: Dict[str, Set[str]] - {query_id: list of possible answers}
        """
        corpus, queries, qrels = BeirDataset._convert_custom_to_beir(df)
        
        # create set of answers for each query
        ground_truth = {
            str(row['id']): list(BeirDataset._get_all_possible_answers(row))
            for _, row in df.iterrows()
        }
        
        return corpus, queries, qrels, ground_truth

    
    @classmethod
    def initialize(cls, hf_hyperlink: str, split: tp.Optional[str]='train') -> 'BeirDataset':
        """
            Builds Beir-like dataset from link to huggingface dataset
        """

        df = pd.DataFrame(load_dataset(hf_hyperlink)[split])
        corpus, queries, qrels, ground_truth = BeirDataset._create_beir_dataset_with_answers(df)

        return cls(corpus, queries, qrels, ground_truth)
