from components import TrainComponents
from sklearn.pipeline import Pipeline
from typing import Callable, Dict, Union
import pandas as pd
from config import settings


class Evaluate(TrainComponents):
    """A composite class representing a evaluation instance of a machine learning pipeline using specified metrics.

    This class calculates evaluation metrics for a given pipeline and evaluation data.
    It inherits from TrainComponents.

    Args:
        metrics (Dict[str, Callable]): A dictionary containing metric names as keys 
            and callable functions as values to compute each metric.

    Attributes:
        metrics (Dict[str, Callable]): Dictionary containing metric names and corresponding metric functions.
    """
    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics

    def _calculate_metrics(self, pipeline: Pipeline, eval_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates the evaluation metrics for a given pipeline and data.

        Args:
            pipeline (Pipeline): The scikit-learn pipeline to evaluate.
            eval_data (pd.DataFrame): The evaluation data.

        Returns:
            Dict[str, float]: Dictionary of metric names and their calculated values.
        """
        result = {}
        y_hat = pipeline.predict(eval_data[settings.TRAIN_FEATURES])
        for metric_name, func in self.metrics.items():
            result[metric_name] = func(y_hat, eval_data[settings.TARGET_FEATURE])
        return result

    def execute(self, data: Pipeline) -> Dict[str, Union[float, Callable[..., object]]]:
        """
        Executes the evaluation process and returns the metrics and pipeline.

        Args:
            data (Dict[str, Union[Pipeline, pd.DataFrame]]): Dictionary containing the pipeline and test data.

        Returns:
            Dict[str, Union[Dict[str, float], Pipeline]]: Dictionary containing the calculated metrics and the pipeline.
        """
        metrics = self._calculate_metrics(pipeline=data["pipeline"], eval_data=data["test_data"])
        return {"metrics": metrics, "pipeline": data["pipeline"]}
