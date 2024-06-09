from typing import Dict, Callable, Any, Union
from train_pipeline.components import TrainComponents
from train_pipeline.components.ml_pipeline.pipeline import MlPipeline
import mlflow
from dynaconf import settings
from datetime import datetime
import hashlib

class MlflowSklearnWriter(TrainComponents):
    """A class for writing models to MLflow.

    This class provides methods to serialize a model and log it as an artifact in MLflow.

    Args:
        BaseWriter: The base class for writers.

    """
    def __init__(self, parameters: Dict[str, Union[int, float, str]], experiment_name) -> None:
        self.parameters = parameters
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name)
        mlflow.set_experiment(experiment_name)

    def _generate_run_id(self):
        today_date = datetime.now()
        return today_date.strftime("%Y%m%d%H%M%S")

    def execute(self, data: Dict[str, Union[float, Callable[..., object]]]) -> None:
        """Write the serialized instance as an artifact to MLflow.

        Args:
            pipeline (MlPipeline): The pipeline to write.

        """
        pipeline = data["pipeline"]
        metrics_dict = data["metrics"]
        with mlflow.start_run(run_name=self._generate_run_id()):
            # Log the model
            mlflow.sklearn.log_model(pipeline, "model")

            # Log parameters and metrics from dictionaries
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(metrics_dict)
        