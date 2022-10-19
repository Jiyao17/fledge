

# FedCLAR implementation, based on Speech Commands Dataset

from multiprocessing import Process, Pipe
from source.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser
from source.node import Trainer, Aggregator
from source.app import App


class FedCLAR(App):
    class Config(App.Config):
        def __init__(self, data_dir: str, 
            client_num: int=350, cluster_threshold: float=0.1,
            batch_size: int=10, lr: float=0.01,
            device: str="cpu",
            ):
            super().__init__(data_dir, client_num, batch_size)
            self.cluster_threshold = cluster_threshold

    def spawn_clients(self):
        # create subsets
        partitioner = SCDatasetPartitionerByUser(self.config.data_dir, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        clients = []
        clients_pipes = []
        for i in range(self.config.client_num):
            parent_conn, child_conn = Pipe()
            clients_pipes.append(parent_conn)
            task = SCTrainerTask(user_subsets[i][0], user_subsets[i][1], 
                self.config.local_epochs, self.config.lr, self.config.batch_size,
                self.config.device
                )
        
            client = Trainer(task, child_conn)
            clients.append(client)
        
        # create the final aggregator
        agg_task = SCAggregatorTask()
        aggregator = Aggregator(agg_task, clients_pipes, None, False)



if __name__ == "__main__":
    config = App.Config("./dataset/raw",)
    pipe_agg, pipe_trainer = Pipe()
    task = SCTrainerTask()
    client = Trainer(task, pipe_trainer)
    aggregator = Aggregator(task, [pipe_agg])