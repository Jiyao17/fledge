

# fk python for this stupid ugly way to import the parent modules
import sys
# project_root = "/home/shallow/projects/fledge/"
project_root = "/home/tuo28237/projects/fledge/"
app_root = project_root + "source/apps/fl/"
sys.path.append(project_root)

import os
# FedCLAR implementation, based on Speech Commands Dataset

import numpy as np
import matplotlib.pyplot as plt
import copy

from source.common.app import TaskType, ArchType, Config, App
from source.common.arch import HFLTrainer, HFLAggregator, HFLCommand
from source.common.measure import *
from source.common.tasks.sc import *

# Classic Federated Learning Architecture

class FLTaskType(TaskType):
    SC = 0 # Speech Commands Recognition
    CIFAR10 = 1 # Image Classification


class FLArchType(ArchType):
    FL_DIRICHLET = 0 # Dirichlet Distributed Federated Learning
    FL_PERSONALIZED = 1 # Personalized Federated Learning


class ArchConfigDrch(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=50, lr: float=0.01,
        device: str="cpu",
        result_dir: str=project_root + "results/iid/",
        data_num_range: tuple=(100, 501), alpha_range: tuple=(100, 100),
        ):
        super().__init__(data_dir, task_type, client_num, batch_size, lr, local_epochs, device, result_dir)
        # self.proc_num = proc_num
        self.global_epochs = global_epochs
        # self.local_epochs = local_epochs

        self.data_num_range = data_num_range
        self.alpha_range = alpha_range


class ArchConfigPer(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=10, lr: float=0.01,
        device: str="cpu",
        result_dir: str=project_root,

        data_num_threshold = 100,
        test_ratio = 0.3,
        ):
        super().__init__(data_dir, task_type, client_num, batch_size, lr, local_epochs, device, result_dir)
        # self.proc_num = proc_num
        self.global_epochs = global_epochs

        self.data_num_threshold = data_num_threshold
        self.test_ratio = test_ratio


class FL(App):

    def __init__(self, ArchType: FLArchType, T):
        
        self.config = copy.deepcopy(config)

        if self.config.task_type == FLTaskType.SC:
            trainset, testset = SCTaskHelper.get_datasets(self.config.data_dir)
            
        self.trainset = trainset
        self.testset = testset

        self.root_aggregator = self.build_structure()
    
    def spawn_clients(self, parent: HFLAggregator=None)-> 'list[HFLTrainer]':
        # create users subsets
        if isinstance(self.config, ArchConfigDrch):
            partitioner = SCDatasetPartitionerDirichlet(self.trainset,
                self.config.client_num, self.config.data_num_range, self.config.alpha_range)
            user_subsets = partitioner.get_subsets()
            user_trainsets = user_subsets
            user_testsets = user_subsets
        elif isinstance(self.config, ArchConfigPer):
            self.partitioner = SCDatasetPartitionerByUser(self.trainset)
            user_subsets = self.partitioner.get_pfl_subsets(
                self.config.data_num_threshold, self.config.test_ratio)
            user_trainsets = [user_subsets[i][0] for i in range(len(user_subsets))]
            user_testsets = [user_subsets[i][1] for i in range(len(user_subsets))]
        partitioner.plot_distributions(
            partitioner.distributions, len(partitioner.distributions), 
            self.config.result_dir + "distributions.png")
        with open(self.config.result_dir + "distributions.txt", "w") as f:
            f.write(str(partitioner.distributions))

            cosine_dis = cosine_deviation(partitioner.distributions)
            plot_devi_by_client(cosine_dis, self.config.result_dir + "distribution_cos_devis.png")
            f.write("\nDistribution Cosine Deviation: \n" + str(cosine_dis))
            
            cosine_diffs = cosine_diff_matrix(partitioner.distributions)
            plot_diff_by_client(cosine_diffs, self.config.result_dir + "distribution_cos_diffs.png")
            f.write("\nDistribution Cosine Difference: \n" + str(cosine_diffs))
        # Spawn clients
        clients: list[HFLTrainer] = []
        for i in range(self.config.client_num):
            trainset = user_trainsets[i]
            testset = user_testsets[i] # test is meaningless for non-personalized fl
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
        def plot_diff_by_iter(diffs_by_iter: 'list[np.ndarray]', result_dir):
            """
            cosine distance between every two clients' gradients
            one picture for each iteration
            @diffs_by_iter: list of n*n matrix, n is the number of clients
            """
            if len(diffs_by_iter) == 0:
                return
            # diffs of clients by iteration
            # get cosine_diffs of each client by iteration
            # for each client
            for j in range(diffs_by_iter[0].shape[0]):
                # current client
                diffs_to_other_clients = []
                for k in range(diffs_by_iter[0].shape[1]):
                    # each other client
                    diffs_to_single_client_by_iter = []
                    for l in range(len(diffs_by_iter)):
                        # each iter
                        diffs_to_single_client_by_iter.append(diffs_by_iter[l][j][k])
                    diffs_to_other_clients.append(diffs_to_single_client_by_iter)
                
                plt.figure()
                for k in range(len(diffs_to_other_clients)):
                    if j != k:
                        plt.plot(range(len(diffs_to_other_clients[k])), diffs_to_other_clients[k], label=f'To Client {k}')
                        plt.legend()
                plt.savefig(result_dir + f"client{j}.png")
                plt.close()

        def plot_devi_by_iter(devis_by_iter: 'list[np.ndarray]', result_file):
            """
            cosine distance between global model and clients' gradients
            one picture for all iteration
            one line for each client
            """
            # deviations of clients by iteration
            # get cosine_devis of each client by iteration
            # for each client
            if len(devis_by_iter) == 0:
                return

            plt.figure()
            for j in range(devis_by_iter[0].shape[0]):
                devis_by_client = []
                for k in range(len(devis_by_iter)):
                    devis_by_client.append(devis_by_iter[k][j])
                plt.plot(range(len(devis_by_client)), devis_by_client, label=f'Client {j}')
            plt.legend()
            plt.savefig(result_file)
            plt.close()
        
        # launch aggregator
        print("Clients data nums: ", self.root_aggregator.children_data_num)
        cosine_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        cosine_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        eucl_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        eucl_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        for i in range(self.config.global_epochs):
            # if i % 5 == 4:
            results = self.root_aggregator.exec_command(HFLCommand.SEND_TRAINER_RESULTS)
            print(f'Epoch {i}, personal accu: {results[0]}, loss: {results[1]}')
            accu, loss = self.root_aggregator.exec_command(HFLCommand.SEND_TEST_RESULTS)
            print(f'Epoch {i}, global accu: {accu}, loss: {loss}')

            self.root_aggregator.exec_command(HFLCommand.UPDATE)

            global_model = self.root_aggregator.task.model
            local_models = [client.task.model for client in self.root_aggregator.children]
            
            cosine_deviations = grads_cosine_deviation(global_model, local_models)
            cosine_diffs = grads_cosine_diff(global_model, local_models)
            cosine_devis_by_iter.append(cosine_deviations)
            cosine_diffs_by_iter.append(cosine_diffs)

            dirs = [self.config.result_dir + "/cos_diffs_by_iter/",
                    self.config.result_dir + f"cos_diffs_by_client/",
                    self.config.result_dir + f"cos_devis_by_client/"]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            plot_diff_by_iter(cosine_diffs_by_iter, self.config.result_dir + "/cos_diffs_by_iter/")
            plot_devi_by_iter(cosine_devis_by_iter, self.config.result_dir + f"cos_devis_by_iter.png")
            plot_diff_by_client(cosine_diffs, self.config.result_dir + f"cos_diffs_by_client/{i}.png")
            plot_devi_by_client(cosine_deviations, self.config.result_dir + f"cos_devis_by_client/{i}.png")

            # euclidean distances of clients by iteration
            # get euclidean distances of each client by iteration
            euclidean_deviations = grads_euclidean_deviation(global_model, local_models)
            euclidean_diffs = grads_euclidean_diff(global_model, local_models)
            eucl_devis_by_iter.append(euclidean_deviations)
            eucl_diffs_by_iter.append(euclidean_diffs)
            
            dirs = [self.config.result_dir + "/euc_diffs_by_iter/",
                    self.config.result_dir + f"euc_diffs_by_client/",
                    self.config.result_dir + f"euc_devis_by_client/"]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            plot_diff_by_iter(eucl_diffs_by_iter, self.config.result_dir+ "euc_diffs_by_iter/")
            plot_devi_by_iter(eucl_devis_by_iter, self.config.result_dir + f"euc_devis_by_iter.png")
            plot_diff_by_client(euclidean_diffs, self.config.result_dir + f"euc_diffs_by_client/{i}.png")
            plot_devi_by_client(euclidean_deviations, self.config.result_dir + f"euc_devis_by_client/{i}.png")




if __name__ == "__main__":
    config_iid = ArchConfigDrch(project_root + "datasets/raw/", FLTaskType.SC, 
        global_epochs=100, local_epochs=5,
        client_num=10, batch_size=50, lr=0.01,
        device="cuda",
        result_dir=app_root + "results/iid/",
        data_num_range=(300, 301), alpha_range=(100000, 100000)
        )

    config_niid = ArchConfigDrch(project_root + "datasets/raw/", FLTaskType.SC,
        global_epochs=100, local_epochs=5,
        client_num=10, batch_size=50, lr=0.01,
        device="cuda",
        result_dir=app_root + "results/noniid/",
        data_num_range=(100, 501), alpha_range=(0.1, 0.1)
        )

    config = config_iid
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    fl = FL(config)
    fl.run()