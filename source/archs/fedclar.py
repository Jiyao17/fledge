
import time
import numpy as np
import enum

from multiprocessing.connection import Connection


from ..node import Trainer, Aggregator
from ..tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser


class Cammand(enum.Enum):
    # Cammands
    CLIENT_SEND_WEIGHT = 0
    CLIENT_UPDATE = 1
    CLIENT_SEND_MODEL = 2
    CLIENT_SET_MODEL = 3
    CLIENT_QUIT = 9

    AGGREGATOR_SEND_WEIGHT = 10
    AGGREGATOR_UPDATE = 11
    AGGREGATOR_SEND_MODEL = 12
    AGGREGATOR_SET_MODEL = 13
    AGGREGATOR_QUIT = 19

class FedCLARTrainer(Trainer):

    def __init__(self, task: SCTrainerTask, pipe: Connection):
        super().__init__(task, pipe)
        self.task = task

    def work_loop(self):
        command = self.parent_pipe.recv()
        while command != Cammand.CLIENT_QUIT:
            if command == Cammand.CLIENT_SEND_WEIGHT:
                self.parent_pipe.send(len(self.task.trainset))
                # print("Client sent weight")
            elif command == Cammand.CLIENT_SET_MODEL:
                self.task.model.load_state_dict(self.parent_pipe.recv())
                # print("Client state dict loaded")
            elif command == Cammand.CLIENT_UPDATE:
                # print("Client training...")
                self.task.train()
                # print("Client training done.")
            elif command == Cammand.CLIENT_SEND_MODEL:
                model = self.task.get_model_by_state_dict()
                self.parent_pipe.send(model)
                # print("Client sent model")
            
            command = self.parent_pipe.recv()


class FedCLARAggregator(Aggregator):

    def __init__(self, task: SCAggregatorTask, epochs: int, device: str, trainer_pipes: 'list[Connection]', parent_pipe: Connection = None, verbose=False):
        super().__init__(task, epochs, device, trainer_pipes, parent_pipe, verbose)
        self.task = task

    def init_params(self):
        # must call this function after all trainers procs started and before work_loop
        for i, pipe in enumerate(self.pipes):
            pipe.send(Cammand.CLIENT_SEND_WEIGHT)
            self.weights[i] = pipe.recv()


    def work_loop(self):

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
            self.final_aggregator_work_loop()


    def final_aggregator_work_loop(self):
        # distribute trainers models
        for i, pipe in enumerate(self.pipes):
            pipe.send(Cammand.CLIENT_SET_MODEL)
            pipe.send(self.task.model.state_dict())
        # self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # send training command to all trainers
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_UPDATE)
        # reqeust models
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_SEND_MODEL)
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # wait for trainers' response
        while not np.all(self.response_list):
            for i, pipe in enumerate(self.pipes):
                if self.response_list[i] == False and pipe.poll(5): #  
                    self.update_list[i] = pipe.recv()
                    self.response_list[i] = True
        # aggregate
        self.task.aggregate(self.update_list, self.weights/np.sum(self.weights))

    def report(self):
        pass
