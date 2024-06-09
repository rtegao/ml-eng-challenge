from train_pipeline.components import TrainComponents
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Sequence, Dict, Union, Callable
import pandas as pd


class MlPipeline(TrainComponents):
    def __init__(self, steps: Sequence[Tuple], features: List[str], target: str):
        self.steps = steps
        self.features = features
        self.target = target

    def _define_pipeline(self):
        return Pipeline(self.steps)

    def execute(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Union[str, Callable]]:
        pipeline = self._define_pipeline()
        pipeline.fit(
            X=data["train_data"][self.features],
            y=data["train_data"][self.target])
        data["pipeline"] = pipeline
        return data
