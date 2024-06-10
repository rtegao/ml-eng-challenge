from components import TrainComponents
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Sequence, Dict, Union, Callable
import pandas as pd


class MlPipeline(TrainComponents):
    """A composite class representing a machine learning pipeline, for training and executing a series of steps.

    Args:
        steps (Sequence[Tuple]): A sequence of tuples where each tuple contains a name 
            and an estimator/transformer to be applied in the pipeline.
        features (List[str]): A list of feature column names to be used in the pipeline.
        target (str): The target column name.

    Attributes:
        steps (Sequence[Tuple]): Stores the steps for the pipeline.
        features (List[str]): Stores the list of features.
        target (str): Stores the target column name.
    """
    def __init__(self, steps: Sequence[Tuple], features: List[str], target: str):
        self.steps = steps
        self.features = features
        self.target = target

    def _define_pipeline(self) -> Pipeline:
        """Defines the pipeline with the specified steps.

        Returns:
            Pipeline: An sklearn Pipeline object with the defined steps.
        """
        return Pipeline(self.steps)

    def execute(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Union[str, Callable]]:
        """Executes the pipeline on the training data and stores the fitted pipeline in the data dictionary.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary containing the training data under the key 'train_data'.

        Returns:
            Dict[str, Union[str, Callable]]: The updated data dictionary containing the fitted pipeline 
                under the key 'pipeline'.
        """
        pipeline = self._define_pipeline()
        pipeline.fit(
            X=data["train_data"][self.features],
            y=data["train_data"][self.target]
        )
        data["pipeline"] = pipeline
        return data
