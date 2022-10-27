
from abc import ABC, abstractmethod
import enum

from .data import DatasetPartitioner, DatasetReader


class TaskType(enum.Enum):
    pass


class Config:
    
    def __init__(self, data_dir: str, task_type: TaskType,
        client_num: int=350, 
        batch_size: int=10, lr: float=0.01, local_epochs: int=5,
        device: str="cpu",
        ):
        self.data_dir = data_dir
        self.task_type = task_type
        
        self.client_num = client_num
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs

        self.device = device


class App(ABC):
    """
    Abstract base class for fl architectures, like PFL, HFL, etc.
    """

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def build_structure(self):
        pass

    @abstractmethod
    def run(self):
        pass
