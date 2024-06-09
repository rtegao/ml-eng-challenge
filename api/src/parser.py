from pydantic import BaseModel
from typing import List, Union 
import pandas as pd

class InputData(BaseModel):
    """
    Pydantic model for input data.

    Attributes:
        model_name (str): The name of the model to be used for prediction.
        features (List[str]): List of feature names.
        values (List[Union[str, int, float]]): List of feature values.
    """
    model_name: str
    features: List[str]
    values: List[Union[str, int, float]]

def to_pandas_df(input_data: InputData) -> pd.DataFrame:
    """
    Converts input data to a pandas DataFrame.

    Args:
        input_data (InputData): The input data containing features and values.

    Returns:
        pd.DataFrame: A pandas DataFrame constructed from the input data.
    """
    return pd.DataFrame([input_data.values], columns=input_data.features)