import typing as tp


import rusBeIR.rag.scoring.metrics as metrics_class
import rusBeIR.utils.type_hints as type_hints


class EvaluateGeneration:
    def __init__(self):
        # Default metrics
        self.default_metrics = [
            metrics_class.RougeMetric(),
            metrics_class.BleuMetric(),
            metrics_class.MeteorMetric(),
            metrics_class.CIDErMetric(),
            metrics_class.COMETMetric(),
        ]
    
    def evaluate(self,
                 generated_responses: type_hints.GeneratorResults,
                 reference_responses: type_hints.GroundTruth,
                 source_queries: type_hints.Queries,
                 metrics: tp.Optional[tp.List[metrics_class.BaseMetric]] = None
        ) -> tp.Dict[str, float]:
        """
        Evaluate using default metrics or overloaded set of metrics
        """
        if metrics is None:
            metrics = self.default_metrics

        
        responses_list = []
        reference_list = []
        queries_list = []

        for key in generated_responses.keys():
            responses_list.append(generated_responses[key])
            reference_list.append(reference_responses[key])
            queries_list.append(source_queries[key])
        
        results = {}
        for metric in metrics:

            if not isinstance(metric, metrics_class.BaseMetric):
                 raise ValueError("Metric must inherit from BaseMetric")
            
            metric_results = metric(
                generated_responses=responses_list, 
                reference_responses=reference_list,
                source_queries=queries_list
            )
            results.update(metric_results)
                
        return results
