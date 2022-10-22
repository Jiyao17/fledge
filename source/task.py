
import enum
from copy import deepcopy
from abc import ABC, abstractmethod

from torch import nn
from torch.utils.data.dataset import Dataset


# depict the training structure of fl

class TaskName(enum.Enum):
    pass


class Task(ABC):
    def __init__(self, task_name: TaskName, 
        trainset: Dataset, testset: Dataset, 
        device: str
        ):
        self.task_name = task_name
        self.trainset = trainset
        self.testset = testset
        self.device = device

        self.model: nn.Module = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass


class TrainerTask(Task):
    def __init__(self, task_name: TaskName, 
        trainset: Dataset, testset: Dataset, 
        device: str,
        ):
        super().__init__(task_name, trainset, testset, device)

    def get_model(self):
        return self.model.state_dict()

    def set_model(self, state_dict: dict):
        self.model.load_state_dict(state_dict)


class AggregatorTask(TrainerTask):

    def __init__(self, task_name: TaskName,
        trainset: Dataset, testset: Dataset,
        device: str,
        ):
        super().__init__(task_name, trainset, testset, device)
        
    def aggregate(self, state_dicts: 'list[dict]', weights: 'list[float]'):
        avg_state_dict = deepcopy(state_dicts[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = avg_state_dict[key] * weights[0]

        for key in avg_state_dict.keys():
            for i in range(1, len(state_dicts)):
                avg_state_dict[key] += state_dicts[i][key] * weights[i]
        
        self.model.load_state_dict(avg_state_dict)


