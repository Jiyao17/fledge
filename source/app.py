
from .dataset import DatasetPartitioner, DatasetReader

class App:

    class Config:
        def __init__(self, data_dir: str):
            self.data_dir = data_dir

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        pass

    # def load_datasets(self, config: DatasetPartitioner.Config):
    #     pass

    def spawn_clients(self):
        pass