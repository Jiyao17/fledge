

# FedCLAR implementation, based on Speech Commands Dataset

from ...common.app import TaskType, Config, App
from ...common.arch import HFLTrainer, HFLAggregator, HFLCommand
from ...common.model import model_cosine_similarity
from ...common.tasks.sc import *

import numpy as np


# Classic Federated Learning Architecture

class FLTaskType(TaskType):
    SC = 0 # Speech Commands Recognition
    CIFAR10 = 1 # Image Classification


class FLConfig(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=10, lr: float=0.01,
        device: str="cpu",
        ):
        super().__init__(data_dir, task_type, client_num, batch_size, lr, local_epochs, device)
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
            agg_task = SCAggregatorTask(testset=self.testset)
            aggregator = HFLAggregator(agg_task, self.config.global_epochs,
                self.config.device, children, None)

        for client in children:
            client.parent = aggregator

        return aggregator

    def run(self):
        
        clients = self.spawn_clients()
        aggregator = self.create_aggregator(clients)

        
        # launch aggregator
        aggregator.init_params()
        print("Clients data nums: ", aggregator.children_data_num)
        for i in range(self.config.global_epochs):
            aggregator.exec_command(HFLCommand.UPDATE)

            if i % 5 == 4:
                results = aggregator.exec_command(HFLCommand.SEND_TRAINER_RESULTS)
                print(f'Epoch {i}, personal accu: {results[0]}, loss: {results[1]}')
                accu, loss = aggregator.exec_command(HFLCommand.SEND_TEST_RESULTS)
                print(f'Epoch {i}, global accu: {accu}, loss: {loss}')



if __name__ == "__main__":
    config = FLConfig("./dataset/raw", FLTaskType.SC, 
        clustering_iter=100, local_epochs=5, 
        client_num=100, device="cuda")
    fl = FL(config)
    fl.run()