

# FedCLAR implementation, based on Speech Commands Dataset

from asyncio import Task
import enum

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser
from source.node import Trainer, Aggregator
from source.app import TaskType, Config, App
from source.archs.fedclar import FedCLARAggregator, FedCLARTrainer


class FedCLARTaskType(TaskType):
    SC = 0 # Speech Commands Recognition

class FedCLARConfig(Config):
        def __init__(self, data_dir: str, task_type: FedCLARTaskType = FedCLARTaskType.SC,
            cluster_threshold: float=0.1, clustering_iter: int=10, 
            global_epochs: int=100, cluster_epochs:int =10, local_epochs: int=2,
            client_num: int=350, batch_size: int=10, lr: float=0.01,
            device: str="cpu",
            ):
            super().__init__(data_dir, task_type, client_num, batch_size, lr, device)
            self.clustering_iter = clustering_iter
            self.cluster_threshold = cluster_threshold
            self.global_epochs = global_epochs
            self.cluster_epochs = cluster_epochs
            self.local_epochs = local_epochs


class FedCLAR(App):

    def __init__(self, config: FedCLARConfig):
        self.config = config
        
    def spawn_clients(self)-> tuple[list[FedCLARTrainer], list[Connection]]:
        # create users subsets
        if self.config.task_type == FedCLARTaskType.SC:
            trainset, testset = SCTaskHelper.get_datasets(self.config.data_dir)
            partitioner = SCDatasetPartitionerByUser(trainset, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        clients: list[FedCLARTrainer] = []
        clients_pipes: list[Connection] = []
        for i in range(self.config.client_num):
            parent_conn, child_conn = Pipe()
            clients_pipes.append(parent_conn)
            if self.config.task_type == FedCLARTaskType.SC:
                task = SCTrainerTask(user_subsets[i][0], user_subsets[i][1], 
                    self.config.local_epochs, self.config.lr, self.config.batch_size,
                    self.config.device
                    )
                client = FedCLARTrainer(task, child_conn)
            clients.append(client)

        return clients, clients_pipes
    
    def spawn_aggregator(self, clients_pipes):
        # create the final aggregator
        if self.config.task_type == FedCLARTaskType.SC:
            agg_task = SCAggregatorTask()
            aggregator = FedCLARAggregator(agg_task, self.config.global_epochs,
                self.config.device, clients_pipes, None, False)

        return aggregator

    def clustering(self):
        pass

    def run(self):
        clients, clients_pipes = self.spawn_clients()
        aggregator = self.spawn_aggregator(clients_pipes)

        # start clients
        for client in clients:
            client.work_loop()
        
        # launch aggregator
        for i in range(self.config.clustering_iter):
            aggregator.work_loop()

        self.clustering()

        for i in range(self.config.global_epochs - self.config.clustering_iter):
            aggregator.work_loop()


if __name__ == "__main__":
    config = FedCLARConfig("./dataset/raw", FedCLARTaskType.SC)
    fedclar = FedCLAR(config)
    fedclar.run()