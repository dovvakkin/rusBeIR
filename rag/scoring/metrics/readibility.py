import typing as tp

import textstat
import numpy as np

from rusBeIR.rag.scoring.metrics.base import BaseMetric

class ReadabilityMetrics(BaseMetric):
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        scores = {
            'flesch_reading_ease': [],
            'gunning_fog': [],
            'smog_index': [],
            'coleman_liau_index': []
        }
        
        for text in generated_responses:
            scores['flesch_reading_ease'].append(textstat.flesch_reading_ease(text))
            scores['gunning_fog'].append(textstat.gunning_fog(text))
            scores['smog_index'].append(textstat.smog_index(text))
            scores['coleman_liau_index'].append(textstat.coleman_liau_index(text))
            
        return {k: np.mean(v) for k, v in scores.items()}
