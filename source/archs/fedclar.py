
import time
import numpy as np

from ..node import Trainer, Aggregator, Cammand


class FedCLARTrainer(Trainer):

    def work_loop(self):
        command = self.parent_pipe.recv()
        while command != Cammand.CLIENT_QUIT:

            if command == Cammand.CLIENT_UPDATE:
                self.task.train()
            elif command == Cammand.CLIENT_SEND_MODEL:
                model = self.task.get_model_by_state_dict()
                self.parent_pipe.send(model)
            
            command = self.parent_pipe.recv()


class FedCLARAggregator(Aggregator):

    def work_loop(self):
        for i, pipe in enumerate(self.pipes):
            pipe.send(Cammand.CLIENT_SEND_WEIGHT)
            self.weights[i] = pipe.recv()
        self.weights /= np.sum(self.weights)

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
            for i in range(self.epochs):
                self.final_aggregator_work_loop()


    def final_aggregator_work_loop(self):
        # set trainers models
        for i, pipe in enumerate(self.pipes):
            pipe.send(Cammand.CLIENT_SET_MODEL)
            pipe.send(self.task.model.state_dict())
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # send training command to all trainers
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_UPDATE)
        # reqeust models
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_SEND_MODEL)
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)
        # wait for trainers' response
        while not np.all(self.response_list):
            time.sleep(5)
            for i, pipe in enumerate(self.pipes):
                if self.response_list[i] is False and pipe.poll():
                    self.update_list[i] = pipe.recv()
                    self.response_list[i] = True
        # aggregate
        self.task.aggregate(self.update_list, self.weights/np.sum(self.weights))

    def report(self):
        pass
