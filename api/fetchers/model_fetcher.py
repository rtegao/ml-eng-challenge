import mlflow
from mlflow.entities.experiment import Experiment
from mlflow.exceptions import MlflowException
import pandas as pd
from logging import Logger
from config import settings
from typing import Dict
from sklearn.pipeline import Pipeline

class ModelFetcher:
    def __init__(self, logger: Logger):
        """
        Initializes the ModelFetcher with a logger and loads models.

        Args:
            logger (Logger): Logger instance for logging information.
        """
        self.logger = logger
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        self.models = self.load_models()

    def _get_experiment(self, experiment: str) -> Experiment:
        """
        Retrieves an experiment by its name.

        Args:
            experiment (str): The name of the experiment.

        Returns:
            Experiment: The retrieved experiment.

        Raises:
            Exception: If the experiment is not found.
        """
        experiment = mlflow.get_experiment_by_name(experiment)
        if experiment:
            self.logger.info(f"Loading Model: {experiment.name} loaded")
            return experiment
        raise Exception(f"Error loading Model: {experiment} not found")

    def _search_run(self, experiment: Experiment) -> pd.DataFrame:
        """
        Searches for runs within an experiment.

        Args:
            experiment (Experiment): The experiment to search runs in.

        Returns:
            pd.DataFrame: DataFrame containing experiment runs.

        Raises:
            Exception: If no runs are found for the experiment.
        """
        experiment_runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
        if not experiment_runs.empty:
            self.logger.info(f"Loading Model: Runs from {experiment.name} loaded")
            return experiment_runs
        raise Exception(f"Error loading Model: No available runs for experiment {experiment.name}")

    def _most_recently_model(self, experiment: Experiment) -> str:
        """
        Finds the most recent run for an experiment.

        Args:
            experiment (Experiment): The experiment to find the most recent run for.

        Returns:
            str: The URI of the most recent model run.

        Raises:
            Exception: If the most recent run cannot be found.
        """
        try:
            runs = self._search_run(experiment=experiment)
            most_recent_index = runs["end_time"].idxmax()
            model_run = runs.loc[most_recent_index, "artifact_uri"]
            self.logger.info(f"Loading Model: Most recent run for {experiment.name} found")
            return model_run
        except:
            raise Exception(f"Error loading Model: Unable to find the most recent run for experiment {experiment.name}")

    def load_models(self) -> Dict[str, Pipeline]:
        """
        Loads all available models from MLflow.

        Returns:
            Dict[str, Pipeline]: Dictionary of model names and their corresponding models.

        Raises:
            Exception: If a model fails to load.
        """
        models = {}
        for model in list(settings.AVAILABLE_MODELS):
            try:
                experiment = self._get_experiment(experiment=model)
                model_uri = f"{self._most_recently_model(experiment=experiment)}/model"
                models[model] = mlflow.sklearn.load_model(model_uri)
                self.logger.info(f"Model Loaded: {model}")
            except MlflowException as e:
                raise Exception(f"Failed to load model: {str(e)}")
        return models
    
    def get_model(self, model_name: str) -> Pipeline:
        """
        Retrieves a loaded model by its name.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            Pipeline: The loaded model pipeline.
        """
        return self.models.get(model_name)
