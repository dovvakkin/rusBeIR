from typing import Dict, List, Callable


RetrieverResults =  Dict[str, Dict[str, float]]  # query_id: [doc_id: relevance]
Corpus = Dict[str, Dict[str, str]]  # doc_id: {"title": title, "text": text}
Queries = Dict[str, str]  # query_id: query_text
Qrels = Dict[str, Dict[str, int]]
GroundTruth = Dict[str, List[str]]
QueryPromptMaker = Callable[[str, RetrieverResults, Corpus, Queries], str]  # f(query_id, restriever_result, corpus, queries) -> LLM input
GeneratorResults = Dict[str, str]  # query_id: answer

MetricResponses = List[str]
MetricReferences = List[List[str]]
MetricQueries = List[str]
