
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
import enum
import time

import numpy as np

from .task import Task, TrainerTask, AggregatorTask

class Command():
    pass


class Node(ABC):
    def __init__(self, task: Task):
        self.task = task

    @abstractmethod
    def exec_command(self, command: Command):
        pass

class Trainer(Node):
    def __init__(self, task: TrainerTask, parent: Node,):
        super().__init__(task, )
        self.parent = parent


class Aggregator(Node):
    
    def __init__(self, task: AggregatorTask,
        parent: Node, children: list[Node],
        ):
        super().__init__(task)
        self.parent = parent
        self.children = children
        self.final_aggregator = True if parent is None else False
        # activated trainers in each round
        # self.activation_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        # # responeded trainers in each round
        # self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        # self.update_list = [None] * len(self.children)
        # weights for each trainer
        self.weights = np.zeros((len(self.children), ), dtype=float)

    @abstractmethod
    def update(self):
        pass