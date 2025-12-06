from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Oracle:
    """
    Oracle for validating and evaluating model outputs with customizable metrics.
    Enhanced with multi-round support for iterative workflows.

    Use register_metric to add new metric functions. Metrics should accept (prediction, reference, **kwargs) and return a numeric score.
    Multi-round metrics can access history data to compute trends and improvements.
    """
    def __init__(self, metrics: Optional[Dict[str, Callable]] = None):
        self.metrics: Dict[str, Callable[[Any, Any], float]] = metrics or {}
        self.multi_round_metrics: Dict[str, Callable] = {}

    def register_metric(self, name: str, func: Callable[[Any, Any], float]) -> None:
        """
        Register a new metric.

        Args:
            name: Unique name for the metric
            func: A function taking (prediction, reference, **kwargs) and returning a float
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' is already registered.")
        self.metrics[name] = func

    def register_multi_round_metric(self, name: str, func: Callable) -> None:
        """
        Register a new multi-round metric that can access historical data.

        Args:
            name: Unique name for the metric
            func: A function taking (history, reference, current_iteration, **kwargs) and returning a float
                  where history is a dict with 'prompts', 'outputs', 'scores' lists
        """
        if name in self.multi_round_metrics:
            raise ValueError(f"Multi-round metric '{name}' is already registered.")
        self.multi_round_metrics[name] = func

    def unregister_metric(self, name: str) -> None:
        """
        Unregister an existing metric by name.
        """
        if name in self.metrics:
            del self.metrics[name]
        elif name in self.multi_round_metrics:
            del self.multi_round_metrics[name]
        else:
            raise KeyError(f"Metric '{name}' is not registered.")

    def list_metrics(self) -> List[str]:
        """
        Return a list of registered metric names.
        """
        return list(self.metrics.keys()) + list(self.multi_round_metrics.keys())

    def list_single_round_metrics(self) -> List[str]:
        """
        Return a list of single-round metric names.
        """
        return list(self.metrics.keys())

    def list_multi_round_metrics(self) -> List[str]:
        """
        Return a list of multi-round metric names.
        """
        return list(self.multi_round_metrics.keys())

    def compute(
        self,
        prediction: Any,
        reference: Any,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute specified metrics on a single example.

        Args:
            prediction: Model output
            reference: Ground truth or expected output
            metrics: List of metric names to compute. If None, compute all registered single-round metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to computed score
        """
        to_compute = metrics or self.list_single_round_metrics()
        results: Dict[str, float] = {}
        for name in to_compute:
            if name not in self.metrics:
                if name in self.multi_round_metrics:
                    raise ValueError(f"Metric '{name}' is a multi-round metric. Use compute_with_history() instead.")
                else:
                    raise KeyError(f"Metric '{name}' not registered.")
            func = self.metrics[name]
            try:
                score = func(prediction, reference, **kwargs)
            except TypeError:
                score = func(prediction, reference)
            results[name] = score
        return results

    def compute_with_history(
        self,
        prediction: Any,
        reference: Any,
        history: Dict[str, List[Any]],
        current_iteration: int = 1,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute metrics including multi-round metrics that can access historical data.

        Args:
            prediction: Current model output
            reference: Ground truth or expected output
            history: Dictionary containing 'prompts', 'outputs', 'scores' lists from previous iterations
            current_iteration: Current iteration number
            metrics: List of metric names to compute. If None, compute all registered metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to computed score
        """
        to_compute = metrics or self.list_metrics()
        results: Dict[str, float] = {}
        
        single_round_metrics = [m for m in to_compute if m in self.metrics]
        if single_round_metrics:
            single_results = self.compute(prediction, reference, single_round_metrics, **kwargs)
            results.update(single_results)
        
        multi_round_metrics = [m for m in to_compute if m in self.multi_round_metrics]
        for name in multi_round_metrics:
            func = self.multi_round_metrics[name]
            try:
                score = func(history, reference, current_iteration, prediction=prediction, **kwargs)
            except TypeError:
                score = func(history, reference, current_iteration)
            results[name] = score
        
        return results

    def evaluate_batch(
        self,
        predictions: List[Any],
        references: List[Any],
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Compute metrics on a batch of examples.

        Args:
            predictions: List of model outputs
            references: List of ground truths
            metrics: List of metric names to compute. If None, compute all registered single-round metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to list of scores per example
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        batch_results: Dict[str, List[float]] = {m: [] for m in (metrics or self.list_single_round_metrics())}
        for pred, ref in zip(predictions, references):
            single = self.compute(pred, ref, metrics, **kwargs)
            for name, score in single.items():
                batch_results[name].append(score)
        return batch_results

    def compute_trend_metrics(self, history: Dict[str, List[Any]], metric_name: str) -> Dict[str, float]:
        """
        Compute trend-based metrics from historical scores.

        Args:
            history: Dictionary containing historical data
            metric_name: Name of the metric to analyze trends for

        Returns:
            Dict with trend metrics like improvement_rate, consistency, etc.
        """
        if not history.get("scores"):
            return {}
        
        scores = []
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict) and metric_name in score_dict:
                scores.append(score_dict[metric_name])
        
        if len(scores) < 2:
            return {"trend_available": 0.0}
        
        scores = np.array(scores)
        
        trend_metrics = {
            "improvement_rate": float(scores[-1] - scores[0]) / len(scores),
            "total_improvement": float(scores[-1] - scores[0]),
            "consistency": 1.0 - float(np.std(scores)) if len(scores) > 1 else 1.0,
            "best_score": float(np.max(scores)),
            "worst_score": float(np.min(scores)),
            "average_score": float(np.mean(scores)),
            "is_improving": float(scores[-1] > scores[0]) if len(scores) > 1 else 0.0,
            "monotonic_improvement": float(all(scores[i] <= scores[i+1] for i in range(len(scores)-1))),
        }
        
        return trend_metrics

