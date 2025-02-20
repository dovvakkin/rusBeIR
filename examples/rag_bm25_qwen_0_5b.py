from rusBeIR.utils.format_datasets import BeirDataset
import json

from llmtf.model import HFModel
from rusBeIR.rag.generator.llmtf_generator import LLMTFGenerator

from rusBeIR.rag.rag_pipeline.rag import RAG
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.retrieval.models.dummyQrelsRetriever import DummyQrelsRetriever

from rusBeIR.utils.type_hints import RetrieverResults, Corpus, Queries

from rusBeIR.rag.scoring.evaluation import EvaluateGeneration
from rusBeIR.rag.scoring.metrics import RougeMetric

from rusBeIR.utils.processors import (
    TextProcessor,
    NatashaPunctuationRemover,
    NatashaStopWordsRemover,
    NatashaLemmatizer,
    LowerCaseProcessor
)

def my_prompt_maker(
    query_id: str,
    retriever_results: RetrieverResults,
    corpus: Corpus,
    queries: Queries
) -> str:
    query_doc_ids = [i[0] for i in sorted(retriever_results[query_id].items(), key=lambda x: x[1], reverse=True)]
    query_text = queries[query_id]

    # hack: python3.10 does not support backslash in f-string
    double_newline = '\n\n'
    
    return f"""
Ты получишь на вход тексты, на основе которых нужно ответить на вопрос, который будет в конце. Не повторяй их, сразу отвечай на вопрос

Тексты:
{double_newline.join([corpus[doc_id]["text"] for doc_id in query_doc_ids])}

Вопрос: {query_text}
    """

dataset = BeirDataset.initialize('bearberry/sberquadqa')

hostname = "localhost"
index_name = "scifact"
initialize = True
number_of_shards = 1

bm25_retriever = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
# retriever that always return qrels to eval rag generator with best possible data
dummy_qrels_retriever = DummyQrelsRetriever(dataset.qrels)


generator_model = HFModel(device_map='cuda:0')
generator_model.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
qwen = LLMTFGenerator(generator_model)

bm25_rag = RAG(bm25_retriever, qwen)
qrels_rag = RAG(dummy_qrels_retriever, qwen)

mini_queries = {q: dataset.queries[q] for q, _ in zip(dataset.queries, range(15))}
bm25_retriever_results, bm25_generator_results = bm25_rag.retrieve_and_generate(
    dataset.corpus,
    mini_queries,
    retriever_kwargs_dict={"top_k": 3},
    generator_kwargs_dict={"query_prompt_maker": my_prompt_maker}
)


evl = EvaluateGeneration()
print('bm25 retriever, qwen 0.5b instruct')
print(evl.evaluate(
    bm25_generator_results,
    dataset.ground_truth,
    mini_queries,
    bm25_retriever_results,
    dataset.corpus,
    metrics = [RougeMetric()]
))

print()
print('qrels retriever, qwen 0.5b instruct')
qrels_retriever_results, qrels_generator_results = qrels_rag.retrieve_and_generate(
    dataset.corpus,
    mini_queries,
    retriever_kwargs_dict={"top_k": 3},
    generator_kwargs_dict={"query_prompt_maker": my_prompt_maker, "batch_size": 5, "disable_tqdm": True}
)

with open("dummy_json_1.json", "w") as f:
    json.dump(qrels_generator_results, f)

with open("dummy_gt.json", "w") as f:
    json.dump(dataset.ground_truth, f)

evl = EvaluateGeneration()
print(evl.evaluate(
    qrels_generator_results,
    dataset.ground_truth,
    mini_queries,
    qrels_retriever_results,
    dataset.corpus,
    metrics = [RougeMetric()]
))

# Пример процессинга текстов (дефолтное поведение)
processor = TextProcessor()

qrels_processed_generator_results = {
    k: processor(v) for k, v in zip(
    qrels_generator_results.keys(),
    qrels_generator_results.values()
)}

ground_truth_processed_results = {
    k: processor(v) for k, v in zip(
        dataset.ground_truth.keys(),
        dataset.ground_truth.values()
    )
    if k in qrels_generator_results.keys()
}

print()
print("Результаты при дефолтном процессинге")
print()

print(evl.evaluate(
    qrels_processed_generator_results,
    ground_truth_processed_results,
    mini_queries,
    qrels_retriever_results,
    dataset.corpus,
    metrics = [RougeMetric()]
))

# Пример процессинга текстов (задаваемые вручную функции)

processor = TextProcessor(
    processors=[NatashaPunctuationRemover(), LowerCaseProcessor()]
)

qrels_processed_generator_results = {
    k: processor(v) for k, v in zip(
    qrels_generator_results.keys(),
    qrels_generator_results.values()
)}

ground_truth_processed_results = {
    k: processor(v) for k, v in zip(
        dataset.ground_truth.keys(),
        dataset.ground_truth.values()
    )
    if k in qrels_generator_results.keys()
}

print()
print("Результаты при кастомном процессинге")
print()

print(evl.evaluate(
    qrels_processed_generator_results,
    ground_truth_processed_results,
    mini_queries,
    qrels_retriever_results,
    dataset.corpus,
    metrics = [RougeMetric()]
))