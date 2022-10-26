
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod

import numpy as np

from .task import Task, AggregatorTask

class Node: pass
class Aggregator: pass


class Node(ABC):
    """
    ABC for nodes in fl
    """
    def __init__(self, task: Task, neighbors: list[Node],):
        # everything related to training
        self.task = task

        self.neighbors = neighbors

    @abstractmethod
    def exec_command(self):
        pass
            

class Aggregator(Node):
    """
    1 to multiple aggregator
    Useful in HFL, or conventional FL
    """
    
    def __init__(self, task: AggregatorTask, neighbors: list[Node]
        ):
        super().__init__(task, neighbors)
        # aggregator is the root of a tree structure
        # it must have 2 or more children, and one possible parent
        self.final_aggregator = False
        assert len(neighbors) >= 2, "Aggregator must have at least 2 children."
        if len(neighbors) == 2:
            # no parent, this is the final aggregator
            self.final_aggregator = True
            self.children = neighbors
        else:
            self.final_aggregator = False
            self.parent = neighbors[0]
            self.children = neighbors[1:]

        # activated trainers in each round
        self.activation_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        # responeded trainers in each round
        self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        self.update_list = [None] * len(self.children)
        # weights for each trainer
        self.weights = np.ndarray((len(self.children), ), dtype=float)



