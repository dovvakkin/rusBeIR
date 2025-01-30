from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.datasets.data_loader import GenericDataLoader
from rusBeIR.beir import util
from rusBeIR.rag.generator.hf_generator import HFGenerator
from rusBeIR.rag.rag_pipeline.rag import RAG

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "/tmp/datasets"
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

hostname = "localhost"
index_name = "scifact"
initialize = True
number_of_shards = 1

bm25 = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
qwen = HFGenerator(model_name="Qwen/Qwen2.5-0.5B")

rag = RAG(bm25, qwen)

small_queries = {
    '1': queries['1']
}
retriever_results, generator_results = rag.retrieve_and_generate(corpus, small_queries, {"top_k": 2})

print(generator_results)
