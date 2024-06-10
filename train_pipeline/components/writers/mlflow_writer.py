from typing import Dict, Callable, Any, Union
from components import TrainComponents
from components.ml_pipeline.pipeline import MlPipeline
import mlflow
from config import settings
from datetime import datetime
from logging import Logger

class MlflowSklearnWriter(TrainComponents):
    """A class for writing models to MLflow.

    This class provides methods to serialize a model and log it as an artifact in MLflow.

    Args:
        BaseWriter: The base class for writers.

    """
    def __init__(self, parameters: Dict[str, Union[int, float, str]], experiment_name, logger: Logger) -> None:
        self.parameters = parameters
        self.logger = logger
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name)
            self.logger.info(f"Experiment not found, creating a new experiment")
        mlflow.set_experiment(experiment_name)

    def _generate_run_id(self) -> str:
        """
        Generates a unique identifier for a run based on the current date and time.

        Returns:
            str: A unique identifier for the run, formatted as "%Y%m%d%H%M%S".
        """
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
        