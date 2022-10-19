
from .dataset import DatasetPartitioner, DatasetReader

class App:

    class Config:
        def __init__(self, data_dir: str, 
            client_num: int=350, cluster_threshold: float=0.1,
            batch_size: int=10, 
            ):
            self.data_dir = data_dir
            self.client_num = client_num
            self.batch_size = batch_size

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        pass

    # def load_datasets(self, config: DatasetPartitioner.Config):
    #     pass

    def spawn_clients(self):
        pass