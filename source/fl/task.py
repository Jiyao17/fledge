
import enum
from copy import deepcopy

from torch import nn
from torch.utils.data.dataset import Dataset


# depict the training structure of fl

# class TaskName(enum.Enum):
#     # Task names
#     pass



class TrainerTask():
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset,
        epochs: int, lr: float, batch_size: int,
        device: str
        ):
        self.model = model
        self.dataset = trainset

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device


    def train(self):
        pass

    def set_model_by_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
    
    def report_update(self):
        return self.model.state_dict()

    def save(self, path: str):
        pass



class AggregatorTask():

    def __init__(self, model: nn.Module, dataset: Dataset, ):
        self.model = model
        self.dataset = dataset

    def aggregate(self, ):
        pass

    def save(self, path: str):
        pass

    def update(self, updates: 'list[dict]'):
        state_dict = deepcopy(self.model.state_dict())
        for update in updates:
            for param in self.model.parameters():
                param.data += update[param]