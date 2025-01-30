from rusBeIR.rag.scoring.metrics.base import BaseMetric

from rusBeIR.rag.scoring.metrics.lexical import (
    RougeMetric,
    BleuMetric,
    MeteorMetric,
    CIDErMetric,
    COMETMetric,
)

from rusBeIR.rag.scoring.metrics.semantic import (
    BertScoreMetric,
    SemanticSimilarityMetric,
    CustomSemanticSimilairty,
)

from rusBeIR.rag.scoring.metrics.rag_specific import (
    RAGContextRelevanceMetric,
    SourceAttributionMetric,
    FactualConsistencyMetric,
    AnswerRelevanceMetric,
    HallucinationDetectionMetric,
)

from rusBeIR.rag.scoring.metrics.readibility import (
    ReadabilityMetrics,
)



__all__ = [
    # Базовые классы
    'BaseMetric',

    # Лексические метрики
    'RougeMetric',
    'BleuMetric',
    'MeteorMetric',
    'CIDErMetric',
    'COMETMetric',
    
    # Семантические метрики
    'BertScoreMetric',
    'SemanticSimilarityMetric',
    'CustomSemanticSimilairty',
    
    # RAG-специфичные метрики
    'RAGContextRelevanceMetric',
    'SourceAttributionMetric',
    'FactualConsistencyMetric',
    'AnswerRelevanceMetric',
    'HallucinationDetectionMetric',
    
    # Метрики читаемости
    'ReadabilityMetrics',
]

METRIC_GROUPS = {
    'base': [
        BaseMetric,
    ],
    'lexical': [
        RougeMetric,
        BleuMetric,
        MeteorMetric,
        CIDErMetric,
        COMETMetric,
    ],
    'semantic': [
        BertScoreMetric,
        SemanticSimilarityMetric,
        CustomSemanticSimilairty,
    ],
    'rag_specific': [
        RAGContextRelevanceMetric,
        SourceAttributionMetric,
        FactualConsistencyMetric,
        AnswerRelevanceMetric,
        HallucinationDetectionMetric,
    ],
    'readability': [
        ReadabilityMetrics,
    ]
}