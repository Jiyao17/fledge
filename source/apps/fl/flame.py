

# fk python for this stupid ugly way to import the parent modules
import sys
# project_root = "/home/shallow/projects/fledge/"
project_root = "/home/tuo28237/projects/fledge/"
app_root = project_root + "source/apps/fl/"
sys.path.append(project_root)

# FedCLAR implementation, based on Speech Commands Dataset

from source.common.app import TaskType, Config, App
from source.common.arch import HFLTrainer, HFLAggregator, HFLCommand
from source.common.model import grads_cosine_deviation, grads_cosine_diff
from source.common.tasks.sc import *

import numpy as np
import matplotlib.pyplot as plt
import copy


# Classic Federated Learning Architecture

class FLTaskType(TaskType):
    SC = 0 # Speech Commands Recognition
    CIFAR10 = 1 # Image Classification


class FLConfigDrch(Config):
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


class FL(App):

    def __init__(self, config: FLConfigDrch):
        self.config = copy.deepcopy(config)

        if self.config.task_type == FLTaskType.SC:
            trainset, testset = SCTaskHelper.get_datasets(self.config.data_dir)
            
        self.trainset = trainset
        self.testset = testset

        self.root_aggregator = self.build_structure()
    
    def spawn_clients(self, parent: HFLAggregator=None)-> 'list[HFLTrainer]':
        # create users subsets
        if self.config.task_type == FLTaskType.SC:
            partitioner = SCDatasetPartitionerDirichlet(self.trainset,
                self.config.client_num, self.config.data_num_range, self.config.alpha_range)
            user_subsets = partitioner.get_subsets()

        SCDatasetPartitionerDirichlet.plot_distributions(
            partitioner.distributions, len(partitioner.distributions), 
            self.config.result_dir + "distributions.png")
        
        # Spawn clients
        clients: list[HFLTrainer] = []
        for i in range(self.config.client_num):
            trainset = user_subsets[i]
            testset = user_subsets[i]
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
        devis_by_iter = []
        diffs_by_iter = []
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
            devis_by_iter.append(cosine_deviations)
            diffs_by_iter.append(cosine_diffs)
            x = []
            y = []
            for j in range(len(cosine_diffs)):
                for k in range(len(cosine_diffs[j])):
                    if j != k:
                        x.append(j)
                        y.append(cosine_diffs[j][k])
            # print(f'Epoch {i}, cosine distances: {cosine_deviations}')
            plt.figure()
            plt.scatter(range(len(cosine_deviations)), cosine_deviations, label=f'Epoch {i}')
            plt.legend()
            plt.savefig(self.config.result_dir + f"cosine_deviation/{i}.png")
            plt.close()
            
            plt.figure()
            plt.scatter(x, y, label=f'Epoch {i}')
            plt.legend()
            plt.savefig(self.config.result_dir + f"cosine_diff/{i}.png")
            plt.close()

            # deviations of clients by iteration
            # get cosine_devis of each client by iteration
            # for each client
            plt.figure()
            for j in range(len(cosine_deviations)):
                devis_by_client = []
                for k in range(len(devis_by_iter)):
                    devis_by_client.append(devis_by_iter[k][j])
                plt.plot(range(len(devis_by_client)), devis_by_client, label=f'Client {j}')
            # plt.legend()
            plt.savefig(self.config.result_dir + f"devis_by_iter.png")
            plt.close()
            
            # diffs of clients by iteration
            # get cosine_diffs of each client by iteration
            # for each client
            for j in range(len(cosine_diffs)):
                # current client
                diffs_to_other_clients = []
                for k in range(len(cosine_diffs)):
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
                        # plt.legend()
                plt.savefig(self.config.result_dir + f"diffs_by_iter/client{j}.png")
                plt.close()



if __name__ == "__main__":
    config = FLConfigDrch(project_root + "datasets/raw/", FLTaskType.SC, 
        global_epochs=100, local_epochs=5,
        client_num=100, batch_size=50, lr=0.01,
        device="cuda",
        result_dir=app_root + "results/noniid/",
        data_num_range=(100, 501), alpha_range=(0.1, 0.1)
        )
    
    fl = FL(config)
    fl.run()