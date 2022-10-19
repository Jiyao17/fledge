

# FedCLAR implementation, based on Speech Commands Dataset

from asyncio import Task
import enum

from multiprocessing import Process, Pipe
from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser
from source.node import Trainer, Aggregator
from source.app import TaskType, Config, App


class FedCLARTaskType(TaskType):
    SC = 0

class FedCLARConfig(Config):
        def __init__(self, data_dir: str, task_type: FedCLARTaskType,
            cluster_threshold: float=0.1, clustering_iter: int=10, 
            client_num: int=350, batch_size: int=10, lr: float=0.01,
            device: str="cpu",
            ):
            super().__init__(data_dir, task_type, client_num, batch_size, lr, device)
            self.clustering_iter = clustering_iter
            self.cluster_threshold = cluster_threshold


class FedCLAR(App):

    def __init__(self, config: FedCLARConfig):
        self.config = config
        
    def spawn_clients(self):
        # create users subsets
        if self.config.task_type == FedCLARTaskType.SC:
            partitioner = SCDatasetPartitionerByUser(self.config.data_dir, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        clients = []
        clients_pipes = []
        for i in range(self.config.client_num):
            parent_conn, child_conn = Pipe()
            clients_pipes.append(parent_conn)
            if self.config.task_type == FedCLARTaskType.SC:
                task = SCTrainerTask(user_subsets[i][0], user_subsets[i][1], 
                    self.config.local_epochs, self.config.lr, self.config.batch_size,
                    self.config.device
                    )
        
            client = Trainer(task, child_conn)
            clients.append(client)

        return clients, clients_pipes
    
    def spawn_aggregator(self, clients_pipes):
        # create the final aggregator
        if self.config.task_type == FedCLARTaskType.SC:
            agg_task = SCAggregatorTask()

        aggregator = Aggregator(agg_task, clients_pipes, None, False)
        return aggregator

    def run(self):
        clients, clients_pipes = self.spawn_clients()
        aggregator = self.spawn_aggregator(clients_pipes)

        # start clients
        for client in clients:
            client.work_loop()
        
        # launch aggregator
        for i in range(self.config.clustering_iter):
            aggregator.work_loop()


if __name__ == "__main__":
    config = App.Config("./dataset/raw",)
    pipe_agg, pipe_trainer = Pipe()
    task = SCTrainerTask()
    client = Trainer(task, pipe_trainer)
    aggregator = Aggregator(task, [pipe_agg])