
import enum

# depict the training structure of fl

# class TaskName(enum.Enum):
#     # Task names
#     pass

class Task:
    class Config:
        def __init__(self, name: str, dataset: str, model: str, epochs: int, batch_size: int):
            self.name = name
            self.dataset = dataset
            self.model = model
            self.epochs = epochs
            self.batch_size = batch_size

    def __init__(self, config: Config):
        self.config = config

    def train(self):
        pass

    def save(self, path: str):
        pass