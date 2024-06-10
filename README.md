# Bain & Company Machine Learning Engineering Challenge

This repository contains the solution to the Machine Learning Engineering Challenge provided by Bain & Company.

## Challenge Overview

The task is to productize the existing Jupyter Notebook, ensuring that coding best practices are followed throughout the process.

### Deliverables:
1. A well-structured pipeline that trains the property valuation model using the provided dataset, ensuring reproducibility and scalability.
2. An API that can receive property information and generate accurate valuation predictions.
3. A README file documenting the project, including instructions for running the pipeline and API, dependencies, and any assumptions or suggestions for improvement.

## Training Pipeline
This repository is a monolith with the necessary materials, and one of its folders contains the desired `train_pipeline`.

To run this pipeline, add the train/test path in `settings.yaml` and run `docker-compose up -d train_pipeline`. This command will automatically create an MLflow container to log the model and metrics (http://localhost:5000/) and run the training pipeline.

### Entrypoint
The entry point of the pipeline is `property_model.py`, where the model is trained.

### SequentialTrainer
To train the model at the entry point, a `SequentialTrainer` was implemented. This approach is an implementation of the composite design pattern that allows the user to easily add and remove new components. In this case, the components are the steps of the ML model training process.

This implementation allows new components to be created by simply extending the `TrainComponents` base class. New entry points can be created faster by leveraging the components that have already been developed.

### Components
The components created for this entry point are:
- **CsvFetcher**: Fetches the provided data from a given path.
- **MlPipeline**: Trains a scikit-learn pipeline with the provided training data.
- **Evaluate**: Evaluates the trained model using the provided test data.
- **MlflowSklearnWriter**: Logs the trained model and metrics in an MLflow instance running in a Docker container.

## API
This repository is a monolith with the necessary materials, and one of its folders contains the desired `api`. A FastAPI app was developed to serve this trained model.

The API fetches the last trained models from MLflow. To load the desired models, add them to `AVAILABLE_MODELS` in `settings.yaml`. The API has a basic security system with an API key, so it's necessary to add the API key in a `.secrets.yaml` file, as shown below:

```yaml
.secrets.yaml
default:
  API_KEY: "123"
```

To run this API is just necessary to run `docker-compose up -d fastapi`. Even if the `train_pipeline` has not been runned, this command will do the following stepd:
- Create the Mlflow container
- Run the `train_pipeline` (model training and log into Mlflow)
- Fetch model from Mlflow
- Expose model

The API swagger can be seen at `http://localhost:8000/docs`, and the examples bellow show how to get predicitons by doing A `POST` requests with curl:
```sh
curl  -X POST \
  'http://0.0.0.0:8000/predict' \
  --header 'Accept: */*' \
  --header 'X-API-Key: <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "model_name": "property_price",
  "features": ["type","sector","net_usable_area","net_area","n_rooms","n_bathroom","latitude","longitude"],
  "values": ["departamento",	"vitacura",	140.0,	170.0,	4.0,	4.0,	-33.40123,	-70.58056]
}'
```
That is a bottleneck in this solution, as the MlFlow is running in a container totally separated from the train_pipeline and from the fastapi. A network bridge was created to connect all the containers, and a hardcoded IP address was used to create the MlFlow container.

If any problem happens with the containers' communication, just run the following command:

` docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <mlflow-container-id>` 

Get the IP address and replace it in the docker-compose.yaml and in the settings.yaml.

# Assumptions
Unit tests and a CI/CD pipeline were not requested for this challenge, so they are not included here.
There was an error in the provided notebook where the target was used as a training feature. This problem was corrected in the `train_pipeline`.
No grid search technique was used. It is assumed that the provided parameters are the most suitable ones.

# Improvements
- Tests and CI/CD pipelines are always necessary. Creating the necessary tests and setting up a GitHub Actions pipeline would be a valuable addition.
- Some code in the entry point of the `train_pipeline` could be injected by an orchestration tool, such as the model parameters.
- With the model parameters provided via an orchestration tool, a grid search could be performed to find the most suitable model using new training and test data.
- Stop usind hardcoded IP for the MlFlow container, setup a DNS. 