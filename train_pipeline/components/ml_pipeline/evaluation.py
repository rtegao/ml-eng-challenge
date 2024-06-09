from train_pipeline.components import TrainComponents
from sklearn.pipeline import Pipeline
from typing import Callable, Dict, Union
import pandas as pd
from dynaconf import settings


class Evaluate(TrainComponents):
    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics

    def _calculate_metrics(self, pipeline: Pipeline, eval_data: pd.DataFrame) -> Dict[str, float]:
        result = {}
        y_hat = pipeline.predict(eval_data[settings.TRAIN_FEATURES])
        for metric_name, func in self.metrics.items():
            result[metric_name] = func(y_hat, eval_data[settings.TARGET_FEATURE])
        return result

    def execute(self, data: Pipeline) -> Dict[str, Union[float, Callable[..., object]]]:
        metrics = self._calculate_metrics(pipeline=data["pipeline"], eval_data=data["test_data"])
        return {"metrics": metrics, "pipeline": data["pipeline"]}
