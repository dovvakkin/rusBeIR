from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
from rusBeIR.retrieval.models.HFTransformers import HFTransformers
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset, Value, Features
from datasets import load_dataset
from collections import defaultdict
from rusBeIR.beir.reranking.models import CrossEncoder
from rusBeIR.beir.reranking import Rerank

# datasets is a list of datasets which are currently included in this benchmark
# elements are {'dataset_name' :(hf_corpus&queries_repo, hf_qrels_repo, split)}

retriever = EvaluateRetrieval()


class DatasetEvaluator:
    def __init__(self, model, k_values=[1, 3, 5, 10, 100]):

        metrics = ['NDCG', 'MAP', 'Recall', 'P', 'MRR']

        self.datasets = {
            'rus-nfcorpus': ('kngrg/rus-nfcorpus', 'kngrg/rus-nfcorpus-qrels', 'test'),
            'rus-arguana': ('kngrg/rus-arguana', 'kngrg/rus-arguana-qrels', 'test'),
            'rus-scifact': ('kngrg/rus-scifact', 'kngrg/rus-scifact-qrels', 'test'),
            'rus-scidocs': ('kngrg/rus-scidocs', 'kngrg/rus-scidocs-qrels', 'test'),
            'rus-trec-covid': ('kngrg/rus-trec-covid', 'kngrg/rus-trec-covid-qrels', 'test'),
            'rus-fiqa': ('kngrg/rus-fiqa', 'kngrg/rus-fiqa-qrels', 'dev'),
            'rus-quora': ('kngrg/rus-quora', 'kngrg/rus-quora-qrels', 'dev'),
            "rus-cqadupstack": ('kngrg/rus-cqadupstack', 'kngrg/rus-cqadupstack-qrels', 'test'),
            'rus-touche': ('kngrg/rus-touche', 'kngrg/rus-touche-qrels', 'test'),
            
            'rus-mmarco': ('kngrg/rus-mmarco-google', 'kngrg/rus-mmarco-qrels', 'dev'),
            'rus-miracl': ('kngrg/rus-miracl', 'kngrg/rus-miracl-qrels', 'dev'),
            'rus-xquad': ('kngrg/rus-xquad', 'kngrg/rus-xquad-qrels', 'dev'),
            'rus-xquad-sentenes': ('kngrg/rus-xquad-sentences', 'kngrg/rus-xquad-sentences-qrels', 'dev'),
            'rus-tydiqa': ('kngrg/rus-tydiqa', 'kngrg/rus-tydiqa-qrels', 'dev'),
            'sberquad-retrieval': ('kngrg/sberquad-retrieval', 'kngrg/sberquad-retrieval-qrels', 'test'),
            'sberquad-retrieval': ('kngrg/sberquad-retrieval', 'kngrg/sberquad-retrieval-qrels', 'validation'),
            'ruscibench-retrieval': ('kngrg/ruSciBench-retrieval', 'kngrg/ruSciBench-retrieval-qrels', 'dev'),
            'ru-facts': ('kngrg/ru-facts', 'kngrg/ru-facts-qrels', 'train'),
            'rubq': ('kngrg/rubq', 'kngrg/rubq-qrels', 'test'),
            'ria-news': ('kngrg/ria-news', 'kngrg/ria-news-qrels', 'test'),
            
            #'rus-mmarco': ('kngrg/rus-mmarco-google', 'kngrg/rus-mmarco-qrels', 'devpp'),
            #'rus-mmarco': ('kngrg/rus-mmarco-google', 'kngrg/rus-mmarco-qrels', 'devqp'),
            'msmarco': ('BeIR/msmarco', 'BeIR/msmarco-qrels', 'validation'),
            'wikifacts-articles': ('kngrg/wikifacts-articles', 'kngrg/wikifacts-articles-qrels', 'dev'),
            'wikifacts-para': ('kngrg/wikifacts-para', 'kngrg/wikifacts-para-qrels', 'dev'),
            'wikifacts-sents': ('kngrg/wikifacts-sents', 'kngrg/wikifacts-sents-qrels', 'dev'), 
            'wikifacts-sliding_para2': ('kngrg/wikifacts-sliding_para2', 'kngrg/wikifacts-sliding_para2-qrels', 'dev'),
            'wikifacts-sliding_para3': ('kngrg/wikifacts-sliding_para3', 'kngrg/wikifacts-sliding_para3-qrels', 'dev'),
            'wikifacts-sliding_para4': ('kngrg/wikifacts-sliding_para4', 'kngrg/wikifacts-sliding_para4-qrels', 'dev'),
            'wikifacts-sliding_para5': ('kngrg/wikifacts-sliding_para5', 'kngrg/wikifacts-sliding_para5-qrels', 'dev'),
            'wikifacts-sliding_para6': ('kngrg/wikifacts-sliding_para6', 'kngrg/wikifacts-sliding_para6-qrels', 'dev'),
           
        }

        self.metrics = metrics
        self.k_values = k_values
        self.results_dir = Path('rusBeIR-results')
        self.model = model

        self.ndcg_sum = dict.fromkeys([f'NDCG@{k}' for k in k_values], 0)
        self.map_sum = dict.fromkeys([f'MAP@{k}' for k in k_values], 0)
        self.recall_sum = dict.fromkeys([f'Recall@{k}' for k in k_values], 0)
        self.precision_sum = dict.fromkeys([f'P@{k}' for k in k_values], 0)
        self.mrr_sum = dict.fromkeys([f'MRR@{k}' for k in k_values], 0)

    def retrieve(self, text_type: str = 'processed_text', results_path=Path('rusBeIR-results')):
        self.results_dir = Path(results_path)
        self.results_dir.mkdir(exist_ok=True)

        for dataset_name, args in tqdm(self.datasets.items(), desc="Processing datasets"):
            print(f"Processing {dataset_name}...")

            out_file = self.results_dir / \
                f"results_{dataset_name}_{args[2]}.json"
            if out_file.exists():
                print(f"File with results for {
                      dataset_name} already exists, skipping...")
                continue

            corpus, queries, _ = HFDataLoader(hf_repo=args[0], hf_repo_qrels=args[1],
                                              streaming=False, keep_in_memory=False, text_type=text_type).load(
                split=args[2])

            if isinstance(self.model, BM25):
                index_name = dataset_name
                self.model = BM25(
                    index_name=index_name, hostname=self.model.config['hostname'], initialize=True)
                retriever = EvaluateRetrieval(
                    retriever=self.model, k_values=self.k_values)
                results = retriever.retrieve(corpus, queries)
            elif isinstance(self.model, HFTransformers):
                corpus_emb = self.model.encode_passages(
                    [doc['text'] for doc in corpus.values()])
                results = self.model.retrieve(
                    queries, corpus_emb, list(corpus.keys()))

            with out_file.open('w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    def rerank(self, model, text_type: str = 'text',
               batch_size: int = 1, results_path='rusBeIR-results'):
        self.results_dir = Path(results_path)
        self.results_rerank_dir = Path(results_path + '-reranked')

        self.results_rerank_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        self.rerank_model = model
        if isinstance(model, CrossEncoder):
            self.rerank_model = Rerank(model, batch_size=batch_size)

        for dataset_name, args in tqdm(self.datasets.items(), desc="Processing datasets"):
            out_file = self.results_rerank_dir / \
                f"results_{dataset_name}_{args[2]}_reranked.json"
            if out_file.exists():
                print(f"File with results for {
                      dataset_name} already exists, skipping...")
                continue

            corpus, queries, _ = HFDataLoader(hf_repo=args[0], hf_repo_qrels=args[1],
                                              streaming=False, keep_in_memory=False, text_type=text_type).load(
                split=args[2])

            result_file = self.results_dir / \
                f'results_{dataset_name}_{args[2]}.json'

            with result_file.open('r', encoding='utf-8') as f:
                results = json.load(f)

            rerank_results = self.rerank_model.rerank(
                corpus, queries, results, top_k=20)

            with out_file.open('w', encoding='utf-8') as f:
                json.dump(rerank_results, f, ensure_ascii=False, indent=4)

    def evaluate(self, results_path=Path('rusBeIR-results'), results_type: str = "default"):
        self.results_dir = Path(results_path)
        retriever = EvaluateRetrieval(k_values=self.k_values)
        processed_datasets = 0

        self.ndcg_sum = dict.fromkeys([f'NDCG@{k}' for k in self.k_values], 0)
        self.map_sum = dict.fromkeys([f'MAP@{k}' for k in self.k_values], 0)
        self.recall_sum = dict.fromkeys(
            [f'Recall@{k}' for k in self.k_values], 0)
        self.precision_sum = dict.fromkeys(
            [f'P@{k}' for k in self.k_values], 0)
        self.mrr_sum = dict.fromkeys([f'MRR@{k}' for k in self.k_values], 0)

        for dataset_name, args in tqdm(self.datasets.items(), desc="Evaluating results"):
            if results_type == 'rerank':
                result_file = self.results_dir / \
                    f"results_{dataset_name}_{args[2]}_reranked.json"
            elif results_type == 'default':
                result_file = self.results_dir / \
                    f"results_{dataset_name}_{args[2]}.json"

            if not result_file.exists():
                print(f"File with results for {dataset_name} does not exist.")
                continue

            with result_file.open('r', encoding='utf-8') as f:
                results = json.load(f)

            qrels_ds = load_dataset(args[1])[args[2]]
            qrels = defaultdict(dict)
            for row in qrels_ds:
                qrels[str(row['query-id'])][str(row['corpus-id'])
                                            ] = int(row['score'])

            ndcg, _map, recall, precision = retriever.evaluate(
                qrels=qrels, results=results, k_values=self.k_values)
            mrr = retriever.evaluate_custom(
                qrels, results, self.k_values, "mrr")

            for k in self.k_values:
                self.ndcg_sum[f'NDCG@{k}'] += ndcg[f'NDCG@{k}']
                self.map_sum[f'MAP@{k}'] += _map[f'MAP@{k}']
                self.recall_sum[f'Recall@{k}'] += recall[f'Recall@{k}']
                self.precision_sum[f'P@{k}'] += precision[f'P@{k}']
                self.mrr_sum[f'MRR@{k}'] += mrr[f'MRR@{k}']

            processed_datasets += 1

        if processed_datasets > 0:
            ndcg_avg = {
                f'NDCG@{k}': self.ndcg_sum[f'NDCG@{k}'] / processed_datasets for k in self.k_values}
            map_avg = {
                f'MAP@{k}': self.map_sum[f'MAP@{k}'] / processed_datasets for k in self.k_values}
            recall_avg = {
                f'Recall@{k}': self.recall_sum[f'Recall@{k}'] / processed_datasets for k in self.k_values}
            precision_avg = {
                f'P@{k}': self.precision_sum[f'P@{k}'] / processed_datasets for k in self.k_values}
            mrr_avg = {
                f'MRR@{k}': self.mrr_sum[f'MRR@{k}'] / processed_datasets for k in self.k_values}

            self.metrics_results = {
                "NDCG": ndcg_avg,
                "MAP": map_avg,
                "Recall": recall_avg,
                "Precision": precision_avg,
                "MRR": mrr_avg
            }

        else:
            print("None of results were retrieved.")
            self.metrics_results = {}

    def print_results(self):
        if not self.metrics_results:
            print("None of metrics were computed.")
            return

        for metric, results in self.metrics_results.items():
            print(f"Average {metric}:")
            for k, val in results.items():
                print(f"{k}: {val}")
            print('\n')
