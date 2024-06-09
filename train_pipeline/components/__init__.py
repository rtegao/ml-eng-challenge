from abc import ABC, abstractmethod
from typing import Any

class TrainComponents(ABC):
    """Abstract base class for machine learning components.

    This class defines the interface for machine learning components that can be part of a training pipeline.

    Methods:
        execute(self, data: Any) -> Any:
            Executes the machine learning component on the input data.

            Args:
                data (Any): The input data to be processed by the machine learning component.

            Returns:
                Any: The output data after processing by the machine learning component.
    """
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Executes the machine learning component on the input data.

        This method should be overridden by subclasses to define the specific behavior of the machine learning component execution.

        Args:
            data (Any): The input data to be processed by the machine learning component.

        Returns:
            Any: The output data after processing by the machine learning component.
        """
        pass