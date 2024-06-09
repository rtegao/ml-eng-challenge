from train_pipeline.components import TrainComponents
import pandas as pd
from typing import Dict

class CsvFetcher(TrainComponents):
    """Concrete implementation of BaseFetcher for fetching data from an local csv file."""
    
    def execute(self, sources: Dict[str, str]) -> pd.DataFrame:
        """Fetches data from a CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the data from the CSV file.
        """
        result = {}
        for dataset_type, source in sources.items():
            result[dataset_type] = pd.read_csv(source) 
        return result
