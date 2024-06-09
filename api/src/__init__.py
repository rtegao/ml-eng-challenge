from fetchers.model_fetcher import ModelFetcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_fetcher = ModelFetcher(logger=logger)

def get_model_fetcher() -> ModelFetcher:
    """
    Returns the model fetcher instance.

    Returns:
        ModelFetcher: The model fetcher instance.
    """
    return model_fetcher