from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from train_pipeline.components.fetchers.csv_fetcher import CsvFetcher
from train_pipeline.components.ml_pipeline.pipeline import MlPipeline
from train_pipeline.components.ml_pipeline.evaluation import Evaluate
from train_pipeline.components.trainers.sequential_trainer import SequentialTrainer
from train_pipeline.components.writers.mlflow_writer import MlflowSklearnWriter

from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from dynaconf import settings

from typing import Callable, Dict, Union


def main(
        eval_metrics: Dict[str, Callable],
        model_parameters: Dict[str, Union[str, float, int]],
        ml_pipeline: MlPipeline,
        experiment_name: str):
    #fetcher
    fetcher = CsvFetcher()
    #evaluation
    evaluate = Evaluate(metrics=eval_metrics)
    writer = MlflowSklearnWriter(parameters=model_parameters, experiment_name=experiment_name)


    #creating train pipeline
    trainer = SequentialTrainer(name=f"{experiment_name}_train_pipeline")
    #adding pipeline steps
    trainer += [fetcher, ml_pipeline, evaluate, writer]
    #executing pipeline
    trainer.execute(data={"train_data": settings.TRAIN_DATA_PATH,
                          "test_data": settings.TEST_DATA_PATH})


if __name__ == "__main__":
    #preprocessing
    ##technique
    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical',
             categorical_transformer,
             settings.CATEGORICAL_COLUMNS)
        ])

    #model
    ##parameters
    model_parameters = {
        "learning_rate": 0.01,
        "n_estimators": 300,
        "max_depth": 5,
        "loss": "absolute_error"
    }
    ##actual model
    model = GradientBoostingRegressor(**model_parameters)

    #pipeline
    ml_pipeline = MlPipeline(
        steps=[
            ('TargetEncoder', preprocessor),
            ('GradientBoostingRegressor', model)],
        features=settings.TRAIN_FEATURES,
        target=settings.TARGET_FEATURE
    )

    #metrics
    eval_metrics = {
        "RMSE": root_mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "MAE": mean_absolute_error,
    }

    main(
        eval_metrics=eval_metrics,
        model_parameters=model_parameters,
        ml_pipeline=ml_pipeline,
        experiment_name="property_price"
    )
