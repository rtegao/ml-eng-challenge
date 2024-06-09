from abc import ABC, abstractmethod
from typing import Any

class BaseWriter(ABC):
    """Abstract base class for writers.

    Defines the interface for writing data.

    Methods:
        write: Abstract method to write data.
    """

    @abstractmethod
    def write(self):
        """
        This method must be implemented by subclasses to write data.
        """
        pass