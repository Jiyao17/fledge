
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod
from copy import deepcopy
import enum


import numpy as np

from .task import Task, AggregatorTask


class Node: pass
class Aggregator: pass
class Command: pass


class Command(enum.Enum):
    # Commands
    pass


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
    General Aggregator
    """
    
    def __init__(self, task: AggregatorTask, neighbors: list[Node],
        final_aggregator: bool = False,
        ):

        super().__init__(task, neighbors)
        self.final_aggregator = final_aggregator

    # @abstractmethod
    # def aggregate(self):
    #     pass


class HFLCammand(Command):
    # Common Cammands
    SEND_DATA_NUM = 0
    UPDATE = 1
    SEND_MODEL = 2
    SET_MODEL = 3
    SEND_TEST_RESULTS = 4
    QUIT = 9


class HFLTrainer(Node):

    def __init__(self, task: Task, parent: Aggregator):
        super().__init__(task, parent)
        self.task: Task

    def exec_command(self, command: HFLCammand, data=None):
        if command == HFLCammand.SEND_DATA_NUM:
            return len(self.task.trainset)
            # print("Client sent weight")
        elif command == HFLCammand.SET_MODEL:
            self.task.model.load_state_dict(data)
            # print("Client state dict loaded")
        elif command == HFLCammand.UPDATE:
            # print("Client training...")
            self.task.update()
            # print("Client training done.")
        elif command == HFLCammand.SEND_MODEL:
            sd = self.task.get_model_state_dict()
            # print("Client sent model")
            return deepcopy(sd)
        elif command == HFLCammand.SEND_TEST_RESULTS:
            # print("Client sent test results")
            return self.task.test()
        else:
            raise NotImplementedError
            

class HFLAggregator(Aggregator):

    def __init__(self, task: AggregatorTask, neighbors: list[HFLTrainer], final_aggregator: bool = False):

        super().__init__(task, None, final_aggregator)
        # aggregator is the root of a tree structure
        # it must have 2 or more children, and one possible parent
        self.task: AggregatorTask
        if self.final_aggregator is False:
            assert len(neighbors) >= 3, "Non final aggregator must have at least 2 children and 1 parent"
            self.parent = neighbors[0]
            self.children = neighbors[1:]
        else:
            self.children = neighbors

        # activated trainers in each round
        self.activation_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        # responeded trainers in each round
        self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)
        self.update_list = [None] * len(self.children)
        # weights for each trainer
        self.children_data_num = np.ndarray((len(self.children), ), dtype=float)
        self.weights = np.ndarray((len(self.children), ), dtype=float)

        self.personal_test_results = np.zeros((len(self.children), 2), dtype=np.float32)

    def init_params(self):
        # must call this function after all trainers procs started and before work_loop
        for i, child in enumerate(self.children):
            data_num = child.exec_command(HFLCammand.SEND_DATA_NUM)
            self.children_data_num[i] = data_num
            self.weights = self.children_data_num / np.sum(self.children_data_num)

    def exec_command(self, command: HFLCammand, data=None):
        if command == HFLCammand.SEND_DATA_NUM:
            return np.sum(self.children_data_num)
        elif command == HFLCammand.SET_MODEL:
            self.task.model.load_state_dict(data)
        elif command == HFLCammand.UPDATE:
            self.update(data)
        elif command == HFLCammand.SEND_MODEL:
            return self.task.get_model_state_dict()
        elif command == HFLCammand.QUIT:
            for child in self.children:
                child.exec_command(HFLCammand.QUIT)
        else:
            raise NotImplementedError

    def update(self, report_personal_test=False):
        # distribute trainers models
        sd = self.task.get_model_state_dict()
        for i, child in enumerate(self.children):
            sd_send = deepcopy(sd)
            child.exec_command(HFLCammand.SET_MODEL, sd_send)
        # self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # send training command to all trainers
        for child in self.children:
            child.exec_command(HFLCammand.UPDATE)
        # reqeust models
        for i, child in enumerate(self.children):

            self.update_list[i] = child.exec_command(HFLCammand.SEND_MODEL)
            self.response_list[i] = True
        # let clients report personal test results
        if report_personal_test:
            self.personal_test_results = np.zeros((len(self.children), 2), dtype=np.float32)
            self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)

            for i, child in enumerate(self.children):
                accu, loss = child.exec_command(HFLCammand.SEND_TEST_RESULTS)
                self.personal_test_results[i][0] = accu
                self.personal_test_results[i][1] = loss
                self.response_list[i] = True
        # aggregate
        avg_model = self.task.model_avg(self.update_list, self.weights)
        self.task.set_model_state_dict(avg_model)

