

import numpy as np

from ..node import Trainer, Aggregator, Cammand


class FedCLARTrainer(Trainer):

    def work_loop(self):
        command = self.parent_pipe.recv()
        while command != Cammand.CLIENT_QUIT:

            if command == Cammand.CLIENT_UPDATE:
                self.task.train()
            elif command == Cammand.CLIENT_REPORT:
                model = self.task.get_model_by_state_dict()
                self.parent_pipe.send(model)
            
            command = self.parent_pipe.recv()


class FedCLARAggregator(Aggregator):

    def work_loop(self):
        if self.final_aggregator is False:
            command = self.parent_pipe.recv()
            while command != Cammand.AGGREGATOR_QUIT:
                if command == Cammand.AGGREGATOR_UPDATE:
                    self.update()
                elif command == Cammand.AGGREGATOR_REPORT:
                    self.report()

                command = self.parent_pipe.recv()
        else:
            for i, pipe in enumerate(self.pipes):
                pipe.send(Cammand.CLIENT_WEIGHT)
                self.weights[i] = pipe.recv()
            self.weights /= np.sum(self.weights)

            while True:
                # set weights
                for i, pipe in enumerate(self.pipes):
                    pipe.send(Cammand.CLIENT_SET_MODEL)
                    pipe.send(self.task.model.state_dict())
                self.update()
                # self.report()
                # time.sleep(1)

    def update(self):
        # reset response list
        self.response_list = np.full((len(self.pipes), ), dtype=bool, fill_value=False)

        # send update command to all trainers
        for pipe in self.pipes:
            pipe.send(Cammand.CLIENT_UPDATE)
            pipe.send(Cammand.CLIENT_REPORT)
        
        # wait for trainers' response
        while not np.all(self.response_list):
            for i, pipe in enumerate(self.pipes):
                if self.response_list[i] is False and pipe.poll():
                    self.update_list[i] = pipe.recv()
                    self.response_list[i] = True
            
            # time.sleep(1)

        self.task.aggregate(self.update_list, self.weights)
