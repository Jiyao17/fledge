
import enum
from copy import deepcopy
from abc import ABC, abstractmethod

from torch import nn
from torch.utils.data.dataset import Dataset


# depict the training structure of fl

# class TaskName(enum.Enum):
#     # Task names
#     pass



class TrainerTask(ABC):
    def __init__(self, trainset: Dataset, testset: Dataset,
        epochs: int, lr: float, batch_size: int,
        device: str
        ):
        self.model: nn.Module = None
        self.trainset = trainset
        self.testset = testset

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    # def test(self):
    #     pass

    # @abstractmethod
    # def get_update(self):
    #     pass

    # @abstractmethod
    def get_model_by_state_dict(self):
        return self.model.state_dict()

    def set_model_by_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
    

    # def save(self, path: str):
    #     pass


class AggregatorTask(ABC):

    def __init__(self, trainset: Dataset=None, testset: Dataset=None,):
        self.trainset = trainset
        self.testset = testset

        self.model: nn.Module = None

    # @abstractmethod
    def aggregate(self, state_dicts: 'list[dict]', weights: 'list[float]'):
        avg_state_dict = deepcopy(state_dicts[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = avg_state_dict[key] * weights[0]

        for key in avg_state_dict.keys():
            for i in range(1, len(state_dicts)):
                avg_state_dict[key] += state_dicts[i][key] * weights[i]
        
        self.model.load_state_dict(avg_state_dict)
        

    # def save(self, path: str):
    #     pass

    # @abstractmethod
    # def update(self, updates: 'list[dict]'):
    #     pass
        # state_dict = deepcopy(self.model.state_dict())
        # for update in updates:
        #     for param in self.model.parameters():
        #         param.data += update[param]

