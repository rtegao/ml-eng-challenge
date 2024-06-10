from components import TrainComponents
from sklearn.pipeline import Pipeline
from typing import Callable, Dict, Union, Sequence
import pandas as pd
from config import settings
from logging import Logger

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
    def __init__(self, metrics: Dict[str, Callable], logger: Logger):
        self.metrics = metrics
        self.logger = logger

    def _calculate_metrics(self, y_hat: Sequence[float], eval_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates the evaluation metrics.

        Args:
            y_hat (Sequence[float]): Model predicitons over evaluation data.
            eval_data (pd.DataFrame): The evaluation data.

        Returns:
            Dict[str, float]: Dictionary of metric names and their calculated values.
        """
        result = {}
        try:
            for metric_name, func in self.metrics.items():
                result[metric_name] = func(y_hat, eval_data[settings.TARGET_FEATURE])
                self.logger.info(f"Metric calculated: {metric_name}")
            return result
        except:
            Exception("Error to calculate metrics")
    
    def _make_predictions(self, pipeline: Pipeline, eval_data: pd.DataFrame) -> Sequence[float]:
        """
        Make predictions using a pipeline object.

        Parameters:
            pipeline (Pipeline): A scikit-learn pipeline object trained for making predictions.
            eval_data (pd.DataFrame): Input data for making predictions. 
                It should be a pandas DataFrame containing the evaluation features.

        Returns:
            Sequence[float]: Predictions made by the pipeline for each data point in the input.
            return pipeline.predict(eval_data[settings.TRAIN_FEATURES])
        """
        try:
            return pipeline.predict(eval_data[settings.TRAIN_FEATURES])
        except ValueError:
            self.logger.error(f"Error performing evaluation predictions")

    def execute(self, data: Pipeline) -> Dict[str, Union[float, Callable[..., object]]]:
        """
        Executes the evaluation process and returns the metrics and pipeline.

        Args:
            data (Dict[str, Union[Pipeline, pd.DataFrame]]): Dictionary containing the pipeline and test data.

        Returns:
            Dict[str, Union[Dict[str, float], Pipeline]]: Dictionary containing the calculated metrics and the pipeline.
        """
        y_hat = self._make_predictions(pipeline=data["pipeline"], eval_data=data["test_data"])
        metrics = self._calculate_metrics(y_hat=y_hat, eval_data=data["test_data"])
        return {"metrics": metrics, "pipeline": data["pipeline"]}
