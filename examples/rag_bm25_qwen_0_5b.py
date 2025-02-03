from rusBeIR.utils.format_datasets import BeirDataset

from llmtf.model import HFModel
from rusBeIR.rag.generator.llmtf_generator import LLMTFGenerator

from rusBeIR.rag.rag_pipeline.rag import RAG
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25

from utils.type_hints import RetrieverResults, Corpus, Queries

from rusBeIR.rag.scoring.evaluation import EvaluateGeneration

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

hostname = "localhost"
index_name = "scifact"
initialize = True
number_of_shards = 1

bm25 = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

generator_model = HFModel(device_map='cuda:0')
generator_model.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
qwen = LLMTFGenerator(generator_model)

rag = RAG(bm25, qwen)

dataset = BeirDataset.initialize('bearberry/sberquadqa')

mini_queries = {q: dataset.queries[q] for q, _ in zip(dataset.queries, range(15))}

retriever_results, generator_results = rag.retrieve_and_generate(
    dataset.corpus,
    mini_queries,
    retriever_kwargs_dict={"top_k": 3},
    generator_kwargs_dict={"query_prompt_maker": my_prompt_maker}
)
print(generator_results)

evl = EvaluateGeneration()
print(evl.evaluate(
    generator_results,
    dataset.ground_truth,
    mini_queries,
    retriever_results,
    dataset.corpus
))  # all metrics by default
