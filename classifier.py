from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    def __init__(self, data: pd.DataFrame, num_layers: int = 2,
                 num_neurons: int = 256, batch_size: int = 64, n_epochs: int = 50) -> None:
        """
        :param data: DataFrame containing the train/test data.
        :param num_layers: number of layer in the model.
        :param num_neurons: number of neurons in the layer.
        :param batch_size: batch size for training.
        :param n_epochs: number of epochs during training.
        """
        self._data = data
        self._num_layers = num_layers
        self._num_nodes = num_neurons
        self._batch_size = batch_size
        self.n_epochs = n_epochs
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def train_and_test(self):
        pass

    @staticmethod
    @abstractmethod
    def save_results(results: pd.DataFrame, path: str, name: str):
        pass
