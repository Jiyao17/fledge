

# FedCLAR implementation, based on Speech Commands Dataset


from torch.multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser
from source.app import TaskType, Config, App
from source.archs.fedclar import FedCLARAggregator, FedCLARTrainer

import numpy as np


class FedCLARTaskType(TaskType):
    SC = 0 # Speech Commands Recognition


class FedCLARConfig(Config):
    def __init__(self,
        data_dir: str, task_type: FedCLARTaskType = FedCLARTaskType.SC,
        cluster_threshold: float=0.1, clustering_iter: int=20, 
        global_epochs: int=100, cluster_epochs:int =10, local_epochs: int=2,
        client_num: int=350, batch_size: int=10, lr: float=0.01,
        device: str="cpu",
        ):
        super().__init__(data_dir, task_type, client_num, batch_size, lr, local_epochs, device)
        # self.proc_num = proc_num
        self.clustering_iter = clustering_iter
        self.cluster_threshold = cluster_threshold
        self.global_epochs = global_epochs
        self.cluster_epochs = cluster_epochs
        self.local_epochs = local_epochs


class FedCLAR(App):

    def __init__(self, config: FedCLARConfig):
        self.config = config

        if self.config.task_type == FedCLARTaskType.SC:
            trainset, testset = SCTaskHelper.get_raw_datasets(self.config.data_dir)
            
        self.trainset = trainset
        self.testset = testset
    
    def spawn_clients(self)-> 'tuple[list[FedCLARTrainer], list[Connection]]':
        def spawn_client(trainset, testset, child_conn: Connection, config: FedCLARConfig):
            task = SCTrainerTask(trainset, testset, 
                config.local_epochs, config.lr, config.batch_size,
                config.device
                )
            # print(f'Client {i} has {len(user_subsets[i][0])} training samples')
            # print(f'Client {i} has {len(user_subsets[i][1])} testing samples')
            client = FedCLARTrainer(task, child_conn)
            client.work_loop()

        # create users subsets
        if self.config.task_type == FedCLARTaskType.SC:
            partitioner = SCDatasetPartitionerByUser(self.trainset, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        client_procs: list[Process] = []
        clients_pipes: list[Connection] = []
        for i in range(self.config.client_num):
            parent_conn, child_conn = Pipe()
            clients_pipes.append(parent_conn)
            if self.config.task_type == FedCLARTaskType.SC:
                proc = Process(target=spawn_client, 
                    args=(user_subsets[i][0], user_subsets[i][1], child_conn, self.config))
                proc.start()
            client_procs.append(proc)

        return client_procs, clients_pipes
    
    def create_aggregator(self, clients_pipes):
        # create the final aggregator
        if self.config.task_type == FedCLARTaskType.SC:
            agg_task = SCAggregatorTask(testset=self.testset)
            aggregator = FedCLARAggregator(agg_task, self.config.global_epochs,
                self.config.device, clients_pipes, None, False)

        return aggregator

    def clustering(self):
        pass

    def run(self):
        # def spawn_client(client: FedCLARTrainer):
        #     client.work_loop()
        
        clients_procs, clients_pipes = self.spawn_clients()
        aggregator = self.create_aggregator(clients_pipes)

        # start clients
        clients_procs: list[Process] = []
        for client_proc in clients_procs:
            if client_proc.is_alive() == False:
                client_proc.start()
        
        # launch aggregator
        aggregator.init_params()
        print("Clients data nums: ", aggregator.weights)
        for i in range(self.config.clustering_iter):
            if i % 5 == 4:
                aggregator.work_loop(True)
                results = np.sum(aggregator.personal_test_results, axis=0) / aggregator.personal_test_results.shape[0]
                print(f'Epoch {i}, personal accu: {results[0]}, loss: {results[1]}')
                accu, loss = aggregator.task.test()
                print(f'Epoch {i}, global accu: {accu}, loss: {loss}')
            else:
                aggregator.work_loop(False)

        self.clustering()

        for i in range(self.config.global_epochs - self.config.clustering_iter):
            aggregator.work_loop()
            accu, loss = aggregator.task.test()
            print(f'Epoch {i + self.config.clustering_iter} accu: {accu}, loss: {loss}')

        # stop clients
        aggregator.stop_all_trainers()

if __name__ == "__main__":
    config = FedCLARConfig("./dataset/raw", FedCLARTaskType.SC, 
        clustering_iter=100, local_epochs=5, 
        client_num=5, device="cuda")
    fedclar = FedCLAR(config)
    fedclar.run()