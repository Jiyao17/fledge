
# depict the hardware structure of fl
# how the devices are connected to each other

from multiprocessing import Process, Queue, 
from multiprocessing.connection import PipeConnection
from copy import deepcopy
import enum

import numpy as np

from task import TrainerTask, AggregatorTask


class Cammand(enum.Enum):
    # Cammands
    CLIENT_UPDATE = 0
    CLIENT_REPORT = 1
    CLIENT_QUIT = 10

    AGGREGATOR_UPDATE = 11
    AGGREGATOR_REPORT = 12
    AGGREGATOR_QUIT = 20


class Trainer():
    def __init__(self, task: TrainerTask, parent_pipe: PipeConnection,):
        # pipe for communication with aggregator
        self.parent_pipe = parent_pipe
        # everything related to training
        self.task = task

    def work_loop(self):
        command = self.parent_pipe.recv()
        while command != Cammand.CLIENT_QUIT:

            if command == Cammand.CLIENT_UPDATE:
                self.task.train()
            elif command == Cammand.CLIENT_REPORT:
                update = self.task.report_update()
                self.parent_pipe.send(update)
            
            command = self.parent_pipe.recv()
            

class Aggregator():
    
    def __init__(self, task: AggregatorTask, trainer_pipes: 'list[PipeConnection]',  
            final_aggregator=False, parent_pipe: PipeConnection=None, verbose=False
            ):
        # everything related to aggregation
        self.task = task

        # pipes for communication with trainers
        self.pipes = trainer_pipes
        self.final_aggregator = final_aggregator
        self.parent_pipe = parent_pipe
        if self.final_aggregator is True:
            assert self.parent_pipe is None, 'Final aggregator shouldn\'t have parent pipe'
        else:
            assert self.parent_pipe is not None, 'Non-final aggregator should have parent pipe'
        self.verbose = verbose

        # activated trainers in each round
        self.activation_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # responeded trainers in each round
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        self.update_list = [None] * len(self.pipes)

    def work_loop(self):
        if self.final_aggregator is False:
            command = self.parent_pipe.recv()
            while command != Cammand.AGGREGATOR_QUIT:
                if command == Cammand.AGGREGATOR_UPDATE:
                    self.update()
                elif command == Cammand.AGGREGATOR_REPORT:
                    self.report()


                command = self.parent_pipe.recv()

    def update(self):
        # reset response list
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)

        # send update command to all trainers
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_UPDATE)
            # pipe.send(Cammand.CLIENT_REPORT)
        
        # wait for trainers' response
        while not np.all(self.response_list):
            for i, pipe in enumerate(self.pipes):
                if self.response_list[i] is False and pipe.poll():
                    self.update_list[i] = pipe.recv()
                    self.response_list[i] = True

        self.task.aggregate(self.update_list)
        


    def report(self):
        state_dict = self.task.report_update()