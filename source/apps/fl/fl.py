

# fk python for this stupid ugly way to import the parent modules
import sys
project_root = "/home/tuo28237/projects/fledge/"
app_root = project_root + "source/apps/fl/"
sys.path.append(project_root)

# FedCLAR implementation, based on Speech Commands Dataset

from source.common.app import TaskType, Config, App
from source.common.arch import HFLTrainer, HFLAggregator, HFLCommand
from source.common.model import grads_cosine_similarity
from source.common.tasks.sc import *

import numpy as np
import matplotlib.pyplot as plt
import copy


# Classic Federated Learning Architecture

class FLTaskType(TaskType):
    SC = 0 # Speech Commands Recognition
    CIFAR10 = 1 # Image Classification


class FLConfig(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=10, lr: float=0.01,
        device: str="cpu",
        result_dir: str=project_root,
        ):
        super().__init__(data_dir, task_type, client_num, batch_size, lr, local_epochs, device, result_dir)
        # self.proc_num = proc_num
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs


class FL(App):

    def __init__(self, config: FLConfig):
        self.config = config

        if self.config.task_type == FLTaskType.SC:
            trainset, testset = SCTaskHelper.get_datasets(self.config.data_dir)
            
        self.trainset = trainset
        self.testset = testset

        self.root_aggregator = self.build_structure()
    
    def spawn_clients(self, parent: HFLAggregator=None)-> 'list[HFLTrainer]':
        # create users subsets
        if self.config.task_type == FLTaskType.SC:
            partitioner = SCDatasetPartitionerByUser(self.trainset, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        clients: list[HFLTrainer] = []
        for i in range(self.config.client_num):
            trainset = user_subsets[i][0]
            testset = user_subsets[i][1]
            if self.config.task_type == FLTaskType.SC:
                task = SCTrainerTask(trainset, testset, 
                    config.local_epochs, config.lr, config.batch_size,
                    config.device
                    )
                client = HFLTrainer(task, parent)
            clients.append(client)

        return clients

    def create_aggregator(self, children: 'list[HFLTrainer]'):
        # create the final aggregator
        if self.config.task_type == FLTaskType.SC:
            agg_task = SCAggregatorTask(None, self.testset,
                self.config.global_epochs, self.config.lr, self.config.batch_size,
                self.config.device
                )
            aggregator = HFLAggregator(agg_task, children, None)

        for client in children:
            client.parent = aggregator

        return aggregator

    def build_structure(self):
        clients = self.spawn_clients()
        aggregator = self.create_aggregator(clients)
        aggregator.init_params()


        return aggregator

    def run(self):
        
        # launch aggregator
        print("Clients data nums: ", self.root_aggregator.children_data_num)
        for i in range(self.config.global_epochs):
            self.root_aggregator.exec_command(HFLCommand.UPDATE)

            # if i % 5 == 4:
            results = self.root_aggregator.exec_command(HFLCommand.SEND_TRAINER_RESULTS)
            print(f'Epoch {i}, personal accu: {results[0]}, loss: {results[1]}')
            accu, loss = self.root_aggregator.exec_command(HFLCommand.SEND_TEST_RESULTS)
            print(f'Epoch {i}, global accu: {accu}, loss: {loss}')

            global_model = self.root_aggregator.task.model
            local_models = [client.task.model for client in self.root_aggregator.children]
            cosine_diffs = grads_cosine_similarity(global_model, local_models)
            print(f'Epoch {i}, cosine distances: {cosine_diffs}')
            plt.plot(cosine_diffs, label=f'Epoch {i}')

            plt.legend()
            plt.savefig(self.config.result_dir + "cosine_distances.png")






if __name__ == "__main__":
    config = FLConfig(project_root + "/datasets/raw/", FLTaskType.SC, 
        global_epochs=100, local_epochs=2,
        client_num=10, device="cuda",
        result_dir=app_root + "results/"
        )
    fl = FL(config)
    fl.run()