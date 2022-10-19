
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
import enum
import time

import numpy as np

from task import TrainerTask, AggregatorTask


class Cammand(enum.Enum):
    # Cammands
    CLIENT_WEIGHT = 0
    CLIENT_UPDATE = 1
    CLIENT_REPORT = 2
    CLIENT_SET_MODEL = 3
    CLIENT_QUIT = 10

    AGGREGATOR_UPDATE = 11
    AGGREGATOR_REPORT = 12
    AGGREGATOR_WEIGHT = 13
    AGGREGATOR_SET_MODEL = 14
    AGGREGATOR_QUIT = 20


class Trainer(ABC):
    def __init__(self, task: TrainerTask, parent_pipe: Connection,):
        # pipe for communication with aggregator
        self.parent_pipe = parent_pipe
        # everything related to training
        self.task = task

    @abstractmethod
    def work_loop(self):
        pass
            

class Aggregator(ABC):
    """
    1 to multiple aggregator
    """
    
    def __init__(self, task: AggregatorTask, trainer_pipes: 'list[Connection]',  
            parent_pipe: Connection=None, verbose=False
            ):
        # everything related to aggregation
        self.task = task

        # pipes for communication with trainers
        self.pipes = trainer_pipes
        self.final_aggregator = True if parent_pipe is None else False
        self.parent_pipe = parent_pipe
        self.verbose = verbose

        # activated trainers in each round
        self.activation_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # responeded trainers in each round
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        self.update_list = [None] * len(self.pipes)
        self.weights = np.ndarray((len(self.pipes), ), dtype=float)

    @abstractmethod
    def work_loop(self):
        pass
