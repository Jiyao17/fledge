
import pickle
import time
import numpy as np
import enum

from copy import deepcopy
import sys

from multiprocessing.connection import Connection

import torch

from ....common.arch import Node, Aggregator
from ..tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser


class HFLCammand(enum.Enum):
    # Cammands
    CLIENT_SEND_DATA_NUM = 0
    CLIENT_UPDATE = 1
    CLIENT_SEND_MODEL = 2
    CLIENT_SET_MODEL = 3
    CLIENT_SEND_TEST_RESULTS = 4
    CLIENT_QUIT = 9

    AGGREGATOR_SEND_WEIGHT = 10
    AGGREGATOR_UPDATE = 11
    AGGREGATOR_SEND_MODEL = 12
    AGGREGATOR_SET_MODEL = 13
    AGGREGATOR_QUIT = 19


class HFLTrainer(Node):

    def __init__(self, task: SCTrainerTask, parent: Aggregator):
        super().__init__(task, parent)
        self.task: SCTrainerTask

    def exec_command(self, command: HFLCammand, data=None):
        if command == HFLCammand.CLIENT_SEND_DATA_NUM:
            return len(self.task.trainset)
            # print("Client sent weight")
        elif command == HFLCammand.CLIENT_SET_MODEL:
            self.task.model.load_state_dict(data)
            # print("Client state dict loaded")
        elif command == HFLCammand.CLIENT_UPDATE:
            # print("Client training...")
            self.task.train()
            # print("Client training done.")
        elif command == HFLCammand.CLIENT_SEND_MODEL:
            sd = self.task.get_model_by_state_dict()
            # print("Client sent model")
            return deepcopy(sd)
        elif command == HFLCammand.CLIENT_SEND_TEST_RESULTS:
            # print("Client sent test results")
            return self.task.test()
            

class FLAggregator(Aggregator):

    def __init__(self, task: SCAggregatorTask, epochs: int, device: str, 
            children: 'list[Node]', parent: Aggregator = None,
            ):
        super().__init__(task, epochs, device, children, parent)
        self.task = task

        self.personal_test_results = np.zeros((len(self.children), 2), dtype=np.float32)

    def init_params(self):
        # must call this function after all trainers procs started and before work_loop
        for i, child in enumerate(self.children):
            weight = child.exec_command(HFLCammand.CLIENT_SEND_DATA_NUM)
            self.weights[i] = weight


    def work_loop(self, report_personal_test=False):

        if self.final_aggregator is False:
            # command = self.parent_pipe.recv()
            # while command != Cammand.AGGREGATOR_QUIT:
            #     if command == Cammand.AGGREGATOR_UPDATE:
            #         self.all_trainers_train()
            #     elif command == Cammand.AGGREGATOR_SEND_MODEL:
            #         self.report()

            #     command = self.parent_pipe.recv()
            pass
        else:
            self.final_aggregator_work_loop(report_personal_test)


    def final_aggregator_work_loop(self, report_personal_test=False):
        # distribute trainers models
        sd = self.task.model.state_dict()
        for i, child in enumerate(self.children):
            sd_send = deepcopy(sd)
            child.exec_command(HFLCammand.CLIENT_SET_MODEL, sd_send)
        # self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # send training command to all trainers
        for child in self.children:
            child.exec_command(HFLCammand.CLIENT_UPDATE)
        # reqeust models
        for i, child in enumerate(self.children):

            self.update_list[i] = child.exec_command(HFLCammand.CLIENT_SEND_MODEL)
            self.response_list[i] = True
        # let clients report personal test results
        if report_personal_test:
            self.personal_test_results = np.zeros((len(self.children), 2), dtype=np.float32)
            self.response_list = np.full((len(self.children), ), dtype=bool, fill_value=False)

            for i, child in enumerate(self.children):
                accu, loss = child.exec_command(HFLCammand.CLIENT_SEND_TEST_RESULTS)
            # wait for trainers' response
                self.personal_test_results[i][0] = accu
                self.personal_test_results[i][1] = loss
                self.response_list[i] = True
        # aggregate
        self.task.model_avg(self.update_list, self.weights/np.sum(self.weights))

    def stop_all_trainers(self):
        for pipe in self.children:
            pipe.exec_command(HFLCammand.CLIENT_QUIT)
        # self.parent_pipe.close()