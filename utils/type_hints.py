from typing import Dict, Callable


RetrieverResults =  Dict[str, Dict[str, float]]  # query_id: [doc_id: relevance]
Corpus = Dict[str, Dict[str, str]]  # doc_id: {"title": title, "text": text}
Queries = Dict[str, str]  # query_id: query_text
QueryPromptMaker = Callable[[str, RetrieverResults, Corpus, Queries], str]  # f(query_id, restriever_result, corpus, queries) -> LLM input
GeneratorResults = Dict[str, str]  # query_id: answer
