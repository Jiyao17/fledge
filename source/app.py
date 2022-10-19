
from abc import ABC, abstractmethod

from .dataset import DatasetPartitioner, DatasetReader

class App(ABC):

    class Config:
        def __init__(self, data_dir: str, 
            client_num: int=350, 
            batch_size: int=10, lr: float=0.01, local_epochs: int=5,
            device: str="cpu",

            ):
            self.data_dir = data_dir
            self.client_num = client_num

            self.batch_size = batch_size
            self.lr = lr
            self.local_epochs = local_epochs

            self.device = device

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        pass

    # def load_datasets(self, config: DatasetPartitioner.Config):
    #     pass

    @abstractmethod
    def spawn_clients(self):
        pass