

# FedCLAR implementation, based on Speech Commands Dataset


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
            trainset, testset = SCTaskHelper.get_datasets(self.config.data_dir)
            
        self.trainset = trainset
        self.testset = testset
    
    def spawn_clients(self, parent: FedCLARAggregator=None)-> 'list[FedCLARTrainer]':
        # create users subsets
        if self.config.task_type == FedCLARTaskType.SC:
            partitioner = SCDatasetPartitionerByUser(self.trainset, None, None, None)
        user_subsets = partitioner.get_pfl_subsets(100, 0.3)
        # Spawn clients
        clients: list[FedCLARTrainer] = []
        for i in range(self.config.client_num):
            trainset = user_subsets[i][0]
            testset = user_subsets[i][1]
            if self.config.task_type == FedCLARTaskType.SC:
                task = SCTrainerTask(trainset, testset, 
                    config.local_epochs, config.lr, config.batch_size,
                    config.device
                    )
                client = FedCLARTrainer(task, parent)
            clients.append(client)

        return clients

    def create_aggregator(self, children: 'list[FedCLARTrainer]'):
        # create the final aggregator
        if self.config.task_type == FedCLARTaskType.SC:
            agg_task = SCAggregatorTask(testset=self.testset)
            aggregator = FedCLARAggregator(agg_task, self.config.global_epochs,
                self.config.device, children, None)

        return aggregator

    def clustering(self):
        pass

    def run(self):
        # def spawn_client(client: FedCLARTrainer):
        #     client.work_loop()
        
        clients = self.spawn_clients()
        aggregator = self.create_aggregator(clients)
        for client in clients:
            client.parent = aggregator
        
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
                print(f'Epoch {i}')
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
        client_num=100, device="cuda")
    fedclar = FedCLAR(config)
    fedclar.run()