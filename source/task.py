
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
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset,
        epochs: int, lr: float, batch_size: int,
        device: str
        ):
        self.model = model
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
    # def report_update(self):
    #     pass
        # return self.model.state_dict()


    def set_model_by_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
    

    def save(self, path: str):
        pass


class AggregatorTask(ABC):

    def __init__(self, model: nn.Module, dataset: Dataset, ):
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def aggregate(self, ):
        pass

    def save(self, path: str):
        pass

    @abstractmethod
    def update(self, updates: 'list[dict]'):
        pass
        # state_dict = deepcopy(self.model.state_dict())
        # for update in updates:
        #     for param in self.model.parameters():
        #         param.data += update[param]


class HFLTrainerTask(TrainerTask):
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset,
        epochs: int, lr: float, batch_size: int,
        device: str
        ):
        super().__init__(model, trainset, testset, epochs, lr, batch_size, device)

    def train(self):
        pass

    def report_update(self):
        pass


class HFLAggregatorTask(AggregatorTask):
    def __init__(self, model: nn.Module, dataset: Dataset, ):
        super().__init__(model, dataset)

    def aggregate(self, ):
        pass

    def update(self, updates: 'list[dict]'):
        pass