
import enum
from copy import deepcopy
from abc import ABC, abstractmethod

from torch import nn
from torch.utils.data.dataset import Dataset


# depict the training structure of fl

# class TaskName(enum.Enum):
#     # Task names
#     pass



class Task(ABC):
    """
    Everthing about model/training
    """
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
    def update(self):
        pass

    @abstractmethod
    def test(self):
        pass

    # @abstractmethod
    # def get_update(self):
    #     pass

    # @abstractmethod
    def get_model_state_dict(self):
        return self.model.state_dict()

    def set_model_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
    

class AggregatorTask(Task):
    """
    Everything about the model on the aggregator side
    """

    def __init__(self, trainset: Dataset, testset: Dataset,
        epochs: int, lr: float, batch_size: int,
        device: str
        ):
        super().__init__(trainset, testset, epochs, lr, batch_size, device)


    def model_avg(self, state_dicts: 'list[dict]', weights: 'list[float]') \
        -> dict:
        avg_state_dict = deepcopy(state_dicts[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = avg_state_dict[key] * weights[0]

        for key in avg_state_dict.keys():
            for i in range(1, len(state_dicts)):
                avg_state_dict[key] += state_dicts[i][key] * weights[i]
        
        return avg_state_dict
        
    def update(self):
        pass


class TaskHelper(ABC):
    AggregatorTaskClass: AggregatorTask
    TrainerTaskClass: Task

    def __init__():
        pass
    
    @abstractmethod
    def get_datasets(self):
        pass