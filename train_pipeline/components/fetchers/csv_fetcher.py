from components import TrainComponents
import pandas as pd
from typing import Dict

class CsvFetcher(TrainComponents):
    """Fetches data from local CSV files.

    CsvFetcher is a concrete implementation of BaseFetcher specifically designed for retrieving 
    data stored in CSV format from local files.

    Attributes:
        Inherits attributes from TrainComponents.

    Methods:
        execute(sources: Dict[str, str]) -> pd.DataFrame:
            Fetches data from one or more CSV files specified in the 'sources' dictionary.

    Example:
        fetcher = CsvFetcher()
        sources = {'train': 'train.csv', 'test': 'test.csv'}
        train_data = fetcher.execute(sources)
    """
    
    def execute(self, sources: Dict[str, str]) -> pd.DataFrame:
        """Fetches data from a CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the data from the CSV file.
        """
        result = {}
        for dataset_type, source in sources.items():
            result[dataset_type] = pd.read_csv(source) 
        return result
