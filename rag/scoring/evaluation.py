import typing as tp


from rusBeIR.rag.scoring.metrics import *


class EvaluateGeneration:
    def __init__(self):
        # Default metrics
        self.default_metrics = [
            RougeMetric(),
            BertScoreMetric(),
            BleuMetric(),
            MeteorMetric(),
            SemanticSimilarityMetric(),
        ]
    
    def evaluate(self, 
                generated_responses: tp.List[str], 
                reference_responses: tp.List[str],
                metrics: tp.Optional[tp.List[BaseMetric]] = None
        ) -> tp.Dict[str, float]:
        """
        Evaluate using default metrics or overloaded set of metrics
        """
        if metrics is None:
            metrics = self.default_metrics
        
        results = {}
        for metric in metrics:

            if not isinstance(metric, BaseMetric):
                 raise ValueError("Metric must inherit from BaseMetric")
            
            metric_results = metric(
                generated_responses, reference_responses
            )
            results.update(metric_results)
                
        return results
