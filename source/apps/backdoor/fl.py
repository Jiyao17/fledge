

# fk python for this stupid ugly way to import the parent modules
import sys
# project_root = "/home/shallow/projects/fledge/"
project_root = "/home/tuo28237/projects/fledge/"
app_root = project_root + "source/apps/backdoor/"
sys.path.append(project_root)

import os
# FedCLAR implementation, based on Speech Commands Dataset

import numpy as np
import matplotlib.pyplot as plt
import copy

from torch.multiprocessing import Process, Queue, Pipe

from source.common.app import TaskType, ArchType, Config, App
from source.common.arch import HFLTrainer, HFLAggregator, HFLCommand
from source.common.measure import *
from source.common.tasks.sc import *
from source.common.tasks.cifar10 import *

from backdoor import get_backdoor_block, BackdoorDataset

# Classic Federated Learning Architecture

class FLTaskType(TaskType):
    SC = 0 # Speech Commands Recognition
    CIFAR10 = 1 # Image Classification


class FLDataDistrType(ArchType):
    FL_DIRICHLET = 0 # Dirichlet Distributed Federated Learning
    FL_PERSONALIZED = 1 # Personalized Federated Learning


class ConfigDrch(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=50, lr: float=0.01,
        device: str="cpu",
        result_dir: str=project_root + "results/iid/",
        data_num_range: tuple=(100, 501), alpha_range: tuple=(100, 100),
        backdoor_client_ratio: float=0.0,
        backdoor_data_ratio: float=0.0,
        ):
        super().__init__(data_dir, task_type, client_num, 
            batch_size, lr, 
            global_epochs, local_epochs, 
            device, result_dir)

        self.data_num_range = data_num_range
        self.alpha_range = alpha_range
        self.backdoor_client_ratio = backdoor_client_ratio
        self.backdoor_data_ratio = backdoor_data_ratio


class ConfigPer(Config):
    def __init__(self, data_dir: str, task_type: FLTaskType = FLTaskType.SC,
        global_epochs: int=100, local_epochs: int=2,
        client_num: int=100, batch_size: int=10, lr: float=0.01,
        device: str="cpu",
        result_dir: str=project_root,

        data_num_threshold = 100,
        test_ratio = 0.3,
        backdoor_client_ratio: float=0.0,
        backdoor_data_ratio: float=0.0,
        ):
        super().__init__(data_dir, task_type, client_num, 
            batch_size, lr, 
            global_epochs, local_epochs, 
            device, result_dir)

        self.data_num_threshold = data_num_threshold
        self.test_ratio = test_ratio
        self.backdoor_ratio = backdoor_client_ratio
        self.backdoor_data_ratio = backdoor_data_ratio


class FL(App):

    def __init__(self, config: Config):
        self.config: ConfigDrch or ConfigPer = copy.deepcopy(config)

        if self.config.task_type == FLTaskType.SC:
            self.task_helper = SCTaskHelper
        elif self.config.task_type == FLTaskType.CIFAR10:
            self.task_helper = CIFAR10TaskHelper

        self.trainset, self.testset = self.task_helper.get_datasets(self.config.data_dir)

        self.root_aggregator = self.build_structure()

    def spawn_clients(self, parent: HFLAggregator=None)-> 'list[HFLTrainer]':
        # create users subsets
        if isinstance(self.config, ConfigDrch):
            if self.config.task_type == FLTaskType.SC:
                partitioner = SCDatasetPartitionerDirichlet(self.trainset,
                    self.config.client_num, self.config.data_num_range, self.config.alpha_range)
            elif self.config.task_type == FLTaskType.CIFAR10:
                partitioner = CIFAR10PartitionerDrichlet(self.trainset,
                    self.config.client_num, self.config.data_num_range, self.config.alpha_range)
            user_subsets = partitioner.get_subsets()
            user_trainsets = user_subsets
            user_testsets = user_subsets
        elif isinstance(self.config, ConfigPer):
            partitioner = SCDatasetPartitionerByUser(self.trainset)
            user_subsets = partitioner.get_pfl_subsets(self.config.client_num,
                self.config.data_num_threshold, self.config.test_ratio)
            user_trainsets = [user_subsets[i][0] for i in range(self.config.client_num)]
            user_testsets = [user_subsets[i][1] for i in range(self.config.client_num)]
        else:
            raise NotImplementedError

        partitioner.plot_distributions(
            partitioner.distributions, len(partitioner.distributions), 
            self.config.result_dir + "distributions.png")
        # with open(self.config.result_dir + "distributions.txt", "w") as f:
        #     f.write(str(partitioner.distributions))

        #     cosine_dis = cosine_deviation(partitioner.distributions)
        #     plot_devi_by_client(cosine_dis, self.config.result_dir + "distribution_cos_devis.png")
        #     f.write("\nDistribution Cosine Deviation: \n" + str(cosine_dis))
            
        #     cosine_diffs = cosine_diff_matrix(partitioner.distributions)
        #     plot_diff_by_client(cosine_diffs, self.config.result_dir + "distribution_cos_diffs.png")
        #     f.write("\nDistribution Cosine Difference: \n" + str(cosine_diffs))
        
        # Spawn clients
        clients: list[HFLTrainer] = []

        # backdoor client dataset
        if self.config.backdoor_client_ratio > 0:
            backdoor = get_backdoor_block()
            backdoor = backdoor.astype(np.float32)
            backdoor_client_num = int(self.config.client_num * self.config.backdoor_client_ratio)
            for i in range(backdoor_client_num):
                backdoor_indices = np.random.choice(len(user_trainsets[i]), int(len(backdoor) * self.config.backdoor_data_ratio))
                user_trainsets[i] = BackdoorDataset(
                    user_trainsets[i], backdoor_indices, backdoor)
                backdoor_indices = range(len(user_testsets[i]))
                user_testsets[i] = BackdoorDataset(
                    user_testsets[i], backdoor_indices, backdoor)

        for i in range(self.config.client_num):
            trainset = user_trainsets[i]
            testset = user_testsets[i] # test is meaningless for non-personalized fl

            task = self.task_helper.TrainerTaskClass(trainset, testset, 
                self.config.local_epochs, self.config.lr, self.config.batch_size,
                self.config.device
                )
            client = HFLTrainer(task, parent)
            clients.append(client)

        return clients

    def create_aggregator(self, children: 'list[HFLTrainer]'):
        # create the final aggregator
        agg_task = self.task_helper.AggregatorTaskClass(None, self.testset,
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
        def plot_diff_by_iter(diffs_by_iter: 'list[np.ndarray]', result_dir, backdoor_ratio=0.0):
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
            backdoor_num = int(diffs_by_iter[0].shape[0] * backdoor_ratio)
            colors  = ["red"] * backdoor_num + ["green"] * (diffs_by_iter[0].shape[0] - backdoor_num)
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
                        plt.plot(range(len(diffs_to_other_clients[k])), diffs_to_other_clients[k],
                            label=f'To Client {k}', color=colors[k])
                        if diffs_by_iter[0].shape[0] <= 10:
                            plt.legend()
                plt.savefig(result_dir + f"client{j}.png")
                plt.close()

        def plot_matrix_by_col(devis_by_iter: 'list[np.ndarray]', result_file, backdoor_ratio=0):
            """
            cosine distance between global model and clients' gradients
            one picture for all iteration
            one line for each client
            """
            # deviations of clients by iteration
            # get cosine_devis of each client by iteration
            # one pic for each client
            if len(devis_by_iter) == 0:
                return

            plt.figure()
            for j in range(devis_by_iter[0].shape[0]):
                devis_by_client = []
                for k in range(len(devis_by_iter)):
                    devis_by_client.append(devis_by_iter[k][j])
                if j < backdoor_ratio*devis_by_iter[0].shape[0]:
                    plt.plot(range(len(devis_by_client)), devis_by_client, label=f'Backdoored', color='red')
                else:
                    plt.plot(range(len(devis_by_client)), devis_by_client, label=f'Honest', color='green')
            if devis_by_iter[0].shape[0] <= 10:
                plt.legend()
            plt.savefig(result_file)
            plt.close()
        
        # launch aggregator
        print("Clients data nums: ", self.root_aggregator.children_data_num)
        update_cosine_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        update_cosine_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        update_eucl_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        update_eucl_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        update_l2norm_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        model_cosine_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        model_cosine_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        model_eucl_devis_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        model_eucl_diffs_by_iter: 'list[np.ndarray]' = [] # list of n*n array
        global_accus_by_iter: 'list[float]' = []
        client_avg_accus_by_iter: 'list[np.ndarray]' = [] # list of n-element vector
        global_model_l2norm_by_iter: 'list[float]' = []
        for i in range(self.config.global_epochs):

            last_global_model = self.root_aggregator.task.model
            self.root_aggregator.exec_command(HFLCommand.UPDATE)

            # if i % 5 == 4:
            client_avg_accu, client_avg_loss = self.root_aggregator.exec_command(HFLCommand.SEND_TRAINER_RESULTS)
            print(f'Epoch {i}, personal accu: {client_avg_accu}, loss: {client_avg_loss}')
            client_avg_accus_by_iter.append(client_avg_accu)
            accu, loss = self.root_aggregator.exec_command(HFLCommand.SEND_TEST_RESULTS)
            print(f'Epoch {i}, global accu: {accu}, loss: {loss}')
            global_accus_by_iter.append(accu)

            plt.figure()
            plt.plot(range(len(global_accus_by_iter)), global_accus_by_iter, label='Global')
            plt.plot(range(len(client_avg_accus_by_iter)), client_avg_accus_by_iter, label='Client Avg')
            plt.legend()
            plt.savefig(self.config.result_dir + f"accu.png")
            plt.close()

            # FLAME: cos diff matrix of local models, L2 norm vec of updates
            global_model = self.root_aggregator.task.model
            global_model_vec = flatten_model(global_model)
            global_model_l2norm = np.linalg.norm(global_model_vec)
            print(f'Global model L2 norm: {global_model_l2norm}')
            local_models = [client.task.model for client in self.root_aggregator.children]
            model_vecs = [flatten_model(model) for model in local_models]
            updates = get_updates(last_global_model, local_models)
            avg_update = np.mean(updates, axis=0)
            print(f'Epoch {i}, avg update L2 norm: {np.linalg.norm(avg_update)}')
            update_l2norms = l2norm(updates) # key point for FLAME
            update_cosine_deviations = cosine_deviation(updates)
            update_cosine_diffs = cosine_diff_matrix(updates)
            update_cosine_devis_by_iter.append(update_cosine_deviations)
            update_cosine_diffs_by_iter.append(update_cosine_diffs)
            update_l2norm_by_iter.append(update_l2norms)

            model_cosine_devis = cosine_deviation(model_vecs)
            model_cosine_diffs = cosine_diff_matrix(model_vecs) # key point for FLAME
            model_cosine_devis_by_iter.append(model_cosine_devis)
            model_cosine_diffs_by_iter.append(model_cosine_diffs)
            global_model_l2norm_by_iter.append(global_model_l2norm)

            dirs = [ self.config.result_dir,
                self.config.result_dir + "update_cos_diffs_by_iter/",
                self.config.result_dir + "update_cos_diffs_by_client/",
                self.config.result_dir + "update_cos_devis_by_client/",
                self.config.result_dir + "model_cos_diffs_by_iter/",
                self.config.result_dir + "model_cos_diffs_by_client/",
                self.config.result_dir + "model_cos_devis_by_client/"]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            plot_vec(global_model_l2norm_by_iter, self.config.result_dir + "global_model_l2norm.png")
            plot_matrix_by_col(update_l2norm_by_iter, self.config.result_dir + "update_l2norm_by_iter.png", self.config.backdoor_client_ratio)
            plot_diff_by_iter(update_cosine_diffs_by_iter, self.config.result_dir + "/update_cos_diffs_by_iter/", self.config.backdoor_client_ratio)
            plot_matrix_by_col(update_cosine_devis_by_iter, self.config.result_dir + f"update_cos_devis_by_iter.png", self.config.backdoor_client_ratio)
            plot_diff_by_client(update_cosine_diffs, self.config.result_dir + f"update_cos_diffs_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_devi_by_client(update_cosine_deviations, self.config.result_dir + f"update_cos_devis_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_diff_by_iter(model_cosine_diffs_by_iter, self.config.result_dir + "/model_cos_diffs_by_iter/", self.config.backdoor_client_ratio)
            plot_matrix_by_col(model_cosine_devis_by_iter, self.config.result_dir + f"model_cos_devis_by_iter.png", self.config.backdoor_client_ratio)
            plot_diff_by_client(model_cosine_diffs, self.config.result_dir + f"model_cos_diffs_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_devi_by_client(model_cosine_devis, self.config.result_dir + f"model_cos_devis_by_client/global_round{i}.png", self.config.backdoor_client_ratio)


            # euclidean distances of clients by iteration
            # get euclidean distances of each client by iteration
            update_euclidean_deviations = euclidean_deviation(updates)
            update_euclidean_diffs = euclidean_diff_matrix(updates)
            update_eucl_devis_by_iter.append(update_euclidean_deviations)
            update_eucl_diffs_by_iter.append(update_euclidean_diffs)

            model_eucl_devis = euclidean_deviation(model_vecs)
            model_eucl_diffs = euclidean_diff_matrix(model_vecs)
            model_eucl_devis_by_iter.append(model_eucl_devis)
            model_eucl_diffs_by_iter.append(model_eucl_diffs)
            
            dirs = [self.config.result_dir + "update_euc_diffs_by_iter/",
                    self.config.result_dir + f"update_euc_diffs_by_client/",
                    self.config.result_dir + f"update_euc_devis_by_client/",
                    self.config.result_dir + "model_euc_diffs_by_iter/",
                    self.config.result_dir + f"model_euc_diffs_by_client/",
                    self.config.result_dir + f"model_euc_devis_by_client/"]

            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            
            plot_diff_by_iter(update_eucl_diffs_by_iter, self.config.result_dir+ "update_euc_diffs_by_iter/", self.config.backdoor_client_ratio)
            plot_matrix_by_col(update_eucl_devis_by_iter, self.config.result_dir + f"update_euc_devis_by_iter.png", self.config.backdoor_client_ratio)
            plot_diff_by_client(update_euclidean_diffs, self.config.result_dir + f"update_euc_diffs_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_devi_by_client(update_euclidean_deviations, self.config.result_dir + f"update_euc_devis_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_diff_by_iter(model_eucl_diffs_by_iter, self.config.result_dir + "model_euc_diffs_by_iter/", self.config.backdoor_client_ratio)
            plot_matrix_by_col(model_eucl_devis_by_iter, self.config.result_dir + f"model_euc_devis_by_iter.png", self.config.backdoor_client_ratio)
            plot_diff_by_client(model_eucl_diffs, self.config.result_dir + f"model_euc_diffs_by_client/global_round{i}.png", self.config.backdoor_client_ratio)
            plot_devi_by_client(model_eucl_devis, self.config.result_dir + f"model_euc_devis_by_client/global_round{i}.png", self.config.backdoor_client_ratio)


config_iid = ConfigDrch(project_root + "datasets/raw/", FLTaskType.SC, 
    global_epochs=100, local_epochs=5,
    client_num=20, batch_size=20, lr=0.01,
    device="cuda",
    result_dir=app_root + "results/iid/",
    data_num_range=(120, 121), alpha_range=(100000, 100000)
    )

config_iid_cifar10 = ConfigDrch(project_root + "datasets/raw/", FLTaskType.CIFAR10, 
    global_epochs=100, local_epochs=5,
    client_num=20, batch_size=50, lr=0.01,
    device="cuda",
    result_dir=app_root + "results/iid_cifar/",
    data_num_range=(500, 501), alpha_range=(100000, 100000)
    )

config_iid_cifar10_backdoor = copy.deepcopy(config_iid_cifar10)
config_iid_cifar10_backdoor.result_dir = app_root + "results/iid_cifar_backdoor/"
config_iid_cifar10_backdoor.backdoor_client_ratio = 0.25

config_niid = ConfigDrch(project_root + "datasets/raw/", FLTaskType.SC,
    global_epochs=100, local_epochs=5,
    client_num=20, batch_size=20, lr=0.01,
    device="cuda",
    result_dir=app_root + "results/noniid/",
    data_num_range=(110, 111), alpha_range=(0.1, 0.1)
    )

config_niid_cifar10 = ConfigDrch(project_root + "datasets/raw/", FLTaskType.CIFAR10,
    global_epochs=100, local_epochs=5,
    client_num=20, batch_size=50, lr=0.01,
    device="cuda",
    result_dir=app_root + "results/noniid_cifar/",
    data_num_range=(500, 501), alpha_range=(0.1, 0.1),
    backdoor_client_ratio=0,
    backdoor_data_ratio=0,
    )

config_niid_cifar10_backdoor = copy.deepcopy(config_niid_cifar10)
config_niid_cifar10_backdoor.result_dir = app_root + "results/noniid_cifar_backdoor/"
config_niid_cifar10_backdoor.backdoor_client_ratio = 0.25
config_niid_cifar10_backdoor.backdoor_data_ratio = 1.0

config_niid_cifar10_backdoor_52 = copy.deepcopy(config_niid_cifar10)
config_niid_cifar10_backdoor_52.client_num = 10
config_niid_cifar10_backdoor_52.result_dir = app_root + "results/noniid_cifar_backdoor_52/"
config_niid_cifar10_backdoor_52.backdoor_client_ratio = 0.2
config_niid_cifar10_backdoor_52.backdoor_data_ratio = 0.5

config_niid_cifar10_backdoor_55 = copy.deepcopy(config_niid_cifar10)
config_niid_cifar10_backdoor_55.client_num = 20
config_niid_cifar10_backdoor_55.result_dir = app_root + "results/noniid_cifar_backdoor_55/"
config_niid_cifar10_backdoor_55.backdoor_client_ratio = 0.2
config_niid_cifar10_backdoor_55.backdoor_data_ratio = 0.2




config_personalized = ConfigPer(project_root + "datasets/raw/", FLTaskType.SC, 
    global_epochs=100, local_epochs=5,
    client_num=20, batch_size=20, lr=0.01,
    device="cuda",
    result_dir=app_root + "results/personalized/",
    data_num_threshold=110, test_ratio=0.2,
    )


def run_test_set(configs: 'list[Config]'):
    for config in configs:
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

        fl = FL(config)
        proc = Process(target=fl.run)
        proc.start()

if __name__ == "__main__":

    # config = config_iid
    # config = config_iid_cifar10
    # config = config_iid_cifar10_backdoor
    # config = config_niid
    config = config_niid_cifar10
    # config = config_niid_cifar10_backdoor
    # config = config_niid_cifar10_backdoor_52
    # config = config_niid_cifar10_backdoor_55
    # config = config_personalized
    configs: 'list[Config]' = [config_iid, config_niid, config_personalized]

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    fl = FL(config)

    fl.run()
