
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from threading import Thread
import enum
import time

import numpy as np

from .task import TrainerTask, AggregatorTask

class Aggregator: pass


class Trainer(ABC):
    def __init__(self, task: TrainerTask, parent: Aggregator,):
        # everything related to training
        self.task = task
        # pipe for communication with aggregator
        self.parent = parent

    @abstractmethod
    def exec_command(self):
        pass
            

class Aggregator(ABC):
    """
    1 to multiple aggregator
    """
    
    def __init__(self, task: AggregatorTask, epochs: int, device: str,
        children: 'list[Trainer]', parent: Aggregator=None, 
        ):
        # everything related to aggregation
        self.task = task
        self.epochs = epochs
        self.device = device

        # pipes for communication with trainers
        self.children = children
        self.final_aggregator = True if parent is None else False
        self.parent = parent

        # activated trainers in each round
        self.activation_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        # responeded trainers in each round
        self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        self.update_list = [None] * len(self.children)
        # weights for each trainer
        self.weights = np.ndarray((len(self.children), ), dtype=float)

    @abstractmethod
    def work_loop(self):
        pass
