import typing as tp
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, generated_responses: tp.List[str], reference_responses: tp.List[str]) -> tp.Dict[str, float]:
        """
        Abstract method to compute metric scores
        
        Args:
            generated_responses: List of generated responses
            reference_responses: List of reference responses
            
        Returns:
            Dictionary with metric scores
        """
        raise NotImplementedError()