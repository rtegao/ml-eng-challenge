from train_pipeline.components import TrainComponents
from typing import Any


class SequentialTrainer(TrainComponents):
    """A composite class representing a sequence of machine learning components.
    
    SequentialTrainer is a composite class that allows chaining multiple machine learning components 
    in a sequential manner for training purposes.

    Args:
        name (str): The name of the sequential trainer.

    Attributes:
        name (str): The name of the sequential trainer.
        components (List[TrainComponents]): List to store the components in the sequence.
    """

    def __init__(self, name: str):
        """Initializes the MLComposite.

        Args:
            name (str): The name of the composite.
        """
        self.name = name
        self.components = []

    def __repr__(self) -> str:
        """Returns a string representation of the MLComposite.

        Returns:
            str: A string representation of the MLComposite.
        """
        components_names = ', '.join(str(comp) for comp in self.components)
        return f'MLComposite(name={self.name}, components=[{components_names}])'

    def __add__(self, component: TrainComponents) -> "SequentialTrainer":
        """Adds a component to the MLComposite.

        Args:
            component (TrainComponents): The component to add.

        Returns:
            SequentialTrainer: The SequentialTrainer object with the added component.
        """
        if isinstance(component, (list, tuple)):
            self.components.extend(component)
        else:
            self.components.append(component)
        return self

    def execute(self, data: Any):
        """Executes the sequence of components in the MLComposite.

        Args:
            data (Any): The input data to be processed by the components.

        Returns:
            Any: The output data after processing by all components in the sequence.
        """
        for component in self.components:
            data = component.execute(data)
        return data
