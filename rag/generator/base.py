from abc import ABC, abstractmethod
from rusBeIR.utils.type_hints import RetrieverResults, GeneratorResults, Corpus, Queries, QueryPromptMaker

def default_prompt_maker(
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
Context:
{double_newline.join([corpus[doc_id]["text"] for doc_id in query_doc_ids])}

Query: {query_text}

Your answer:
    """
    
class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self, 
        retriever_results: RetrieverResults,
        corpus: Corpus,
        queries: Queries,
        query_prompt_maker: QueryPromptMaker,
        **kwargs
    ) -> GeneratorResults:
        pass
