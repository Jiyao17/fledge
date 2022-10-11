

import json

class DatasetPartitioner:

    class Config:
        def __init__(self, dataset: str, num_partitions: int):
            self.dataset = dataset
            self.num_partitions = num_partitions

    def __init__(self, config: Config):
        self.config = config

    def partition(self):
        pass

    def save(self, path: str):
        json.dump(self.config, open(path, 'w'))
        pass

class DatasetReader:
    def __init__(self, path: str):
        self.config = json.load(open(path + '/config.json'))
        pass

    def read(self):
        pass