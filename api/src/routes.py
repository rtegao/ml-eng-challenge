from fastapi import APIRouter, Depends
from fetchers.model_fetcher import ModelFetcher
from src import get_model_fetcher 
from src.parser import InputData, to_pandas_df
from src.security import verify_api_key
from typing import Dict

router = APIRouter()

@router.get("/")
async def read_root() -> Dict[str, str]:
    """
    GET endpoint to return a welcome message.

    Returns:
        Dict[str, str]: A dictionary containing a welcome message.
    """
    return {"message": "Hello World"}

@router.post("/predict")
async def predict(input_data: InputData, model_loader: ModelFetcher = Depends(get_model_fetcher), api_key: str = Depends(verify_api_key)) -> float:
    """
    POST endpoint to return the prediction for the given input data.

    Args:
        input_data (InputData): The input data for the prediction.
        model_loader (ModelFetcher, optional): Dependency to load the model. Defaults to getting the model fetcher.

    Returns:
        float: The prediction result from the model.
    """
    model = model_loader.get_model(input_data.model_name)
    data = to_pandas_df(input_data=input_data)
    return model.predict(data)[0]
