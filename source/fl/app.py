
from .dataset import DatasetPartitioner, DatasetReader

class App:

    class Config:
        pass

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        pass

    def load_datasets(self, config: DatasetPartitioner.Config):
        pass

    def spawn_clients(self):
        pass