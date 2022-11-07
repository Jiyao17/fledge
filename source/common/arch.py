
# depict the hardware structure of fl
# how the devices are connected to each other

from abc import ABC, abstractmethod
from copy import deepcopy
import enum


import numpy as np

from .task import Task, AggregatorTask


class Node: pass
class Command: pass


class Command(enum.Enum):
    # Commands
    pass


class Node(ABC):
    """
    ABC for nodes in fl
    """
    def __init__(self, task: Task, neighbors: 'list[Node]',):
        # everything related to training
        self.task = task

        self.neighbors = neighbors

    @abstractmethod
    def exec_command(self):
        pass


class HFLCommand(Command):
    # Common Cammands
    SEND_DATA_NUM = 0
    UPDATE = 1
    SEND_MODEL = 2
    SET_MODEL = 3
    SEND_TEST_RESULTS = 4
    QUIT = 9

    # aggregator specific commands
    SEND_TRAINER_RESULTS = 10


class HFLAggregator(Node): pass


class HFLTrainer(Node):

    def __init__(self, task: Task, parent: HFLAggregator):
        super().__init__(task, [parent,])
        self.task: Task
        self.parent: HFLAggregator = parent

    def exec_command(self, command: HFLCommand, data=None):
        if command == HFLCommand.SEND_DATA_NUM:
            return len(self.task.trainset)
            # print("Client sent weight")
        elif command == HFLCommand.SET_MODEL:
            self.task.model.load_state_dict(data)
            # print("Client state dict loaded")
        elif command == HFLCommand.UPDATE:
            # print("Client training...")
            self.task.update()
            # print("Client training done.")
        elif command == HFLCommand.SEND_MODEL:
            sd = self.task.get_model_state_dict()
            # print("Client sent model")
            return deepcopy(sd)
        elif command == HFLCommand.SEND_TEST_RESULTS:
            # print("Client sent test results")
            return self.task.test()
        else:
            raise NotImplementedError
            

class HFLAggregator(Node):

    def __init__(self, task: AggregatorTask, 
        children: 'list[HFLTrainer]', parent: HFLAggregator = None,
        ):

        super().__init__(task, children + [parent] )
        self.task: AggregatorTask
        # aggregator is the root of a tree structure
        # it must have 2 or more children, and one possible parent
        self.children = children
        self.parent = parent

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
            data_num = child.exec_command(HFLCommand.SEND_DATA_NUM)
            self.children_data_num[i] = data_num
            self.weights = self.children_data_num / np.sum(self.children_data_num)

    def exec_command(self, command: HFLCommand, data=None):
        if command == HFLCommand.SEND_DATA_NUM:
            return np.sum(self.children_data_num)
        elif command == HFLCommand.SET_MODEL:
            self.task.model.load_state_dict(data)
        elif command == HFLCommand.UPDATE:
            self.update(data)
        elif command == HFLCommand.SEND_MODEL:
            return self.task.get_model_state_dict()
        elif command == HFLCommand.QUIT:
            for child in self.children:
                child.exec_command(HFLCommand.QUIT)
        elif command == HFLCommand.SEND_TEST_RESULTS:
            return self.task.test()
        elif command == HFLCommand.SEND_TRAINER_RESULTS:
            for i, child in enumerate(self.children):
                self.personal_test_results[i] = child.exec_command(HFLCommand.SEND_TEST_RESULTS)
            results = np.sum(self.personal_test_results, axis=0) / self.personal_test_results.shape[0]
            return results
        else:
            raise NotImplementedError

    def update(self, report_personal_test=False):
        # distribute trainers models
        sd = self.task.get_model_state_dict()
        for i, child in enumerate(self.children):
            sd_send = deepcopy(sd)
            child.exec_command(HFLCommand.SET_MODEL, sd_send)
        # self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # send training command to all trainers
        for child in self.children:
            child.exec_command(HFLCommand.UPDATE)
        # reqeust models
        for i, child in enumerate(self.children):

            self.update_list[i] = child.exec_command(HFLCommand.SEND_MODEL)
            self.response_list[i] = True
        # let clients report personal test results
        if report_personal_test:
            self.personal_test_results = np.zeros((len(self.children), 2), dtype=np.float32)
            self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)

            for i, child in enumerate(self.children):
                accu, loss = child.exec_command(HFLCommand.SEND_TEST_RESULTS)
                self.personal_test_results[i][0] = accu
                self.personal_test_results[i][1] = loss
                self.response_list[i] = True
        # aggregate
        avg_model = self.task.model_avg(self.update_list, self.weights)
        self.task.set_model_state_dict(avg_model)

        