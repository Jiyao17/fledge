

import json
import enum

from abc import ABC, abstractmethod
from typing import Sequence

import torch 
import torchvision
from torch.utils.data import Dataset, Subset
import torchvision.transforms as tvtf

import numpy as np
import matplotlib.pyplot as plt


# class RealSubset(Subset):
#     def __init__(self, dataset: Dataset, indices: Sequence[int]):
#         self.dataset = []
#         for i in indices:
#             self.dataset.append(dataset[i])
#         self.indices = range(len(indices))
            

#     def __getitem__(self, idx):
#         if isinstance(idx, list):
#             return [self.dataset[i] for i in idx]
#         return self.dataset[idx]

#     def __len__(self):
#         return len(self.indices)


class DatasetPartitioner(ABC):
    """
    need to be implemented for each dataset
    """

    @staticmethod
    def plot_distributions(distributions: np.ndarray, num: int, filename: str="./distribution.png"):
        
        xaxis = np.arange(num)
        base = np.zeros(shape=(num,))
        for i in range(distributions.shape[1]):
            plt.bar(xaxis, distributions[:,i][0:num], bottom=base)
            base += distributions[:,i][0:num]

        # plt.rc('font', size=16)
        # plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

        # plt.xlabel('Clients', fontsize=20)
        # plt.ylabel('Distribution', fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.grid(True)
        # plt.legend()

        # plt.savefig('no_selection.pdf')
        plt.savefig(filename)
        plt.clf()

    @staticmethod
    def get_cvs(distributions: np.ndarray) -> np.ndarray:
        """
        return value:
        cv: np.ndarray = coefficient of variation
        """

        stds = np.std(distributions, axis=1)
        cvs = stds / np.mean(distributions, axis=1)
        return cvs

    @staticmethod
    def save_dataset(dataset: Dataset, filename: str):
        torch.save(dataset, filename)

    @staticmethod
    def load_dataset(filename: str) -> Dataset:
        return torch.load(filename)

    def __init__(self, dataset: Dataset, subset_num: int=1000, 
            data_num_range: 'tuple[int]'=(10, 50), 
            alpha_range: 'tuple[float, float]'=(0.05, 0.5),
            ):
        self.dataset = dataset
        self.subset_num = subset_num
        # range = (min, max)
        self.data_num_range = data_num_range
        # self.label_type_num = len(self.label_types)
        # self.alpha = [alpha] * self.label_type_num
        self.alpha_range = alpha_range

        self.distributions = None
        self.subsets = None

    @abstractmethod
    def get_label_types(self) -> list:
        pass

    @abstractmethod
    def get_targets(self) -> list:
        pass

    def dataset_categorize(self, dataset: Dataset) -> 'list[list[int]]':
        """
        return value:
        (return list)[i]: list[int] = all indices for category i
        """
        label_types = self.get_label_types(dataset)
        targets = self.get_targets(dataset)

        indices_by_lable = [[] for label in label_types]
        for i, target in enumerate(targets):
            category = label_types.index(target)
            indices_by_lable[category].append(i)

        # subsets = [Subset(dataset, indices) for indices in indices_by_lable]
        return indices_by_lable

    def get_distributions(self):

        label_type_num = len(self.get_label_types(self.dataset))

        subsets_sizes = np.random.randint(self.data_num_range[0], self.data_num_range[1], size=self.subset_num)
        # print("subset_size: ", subsets_sizes[:15])
        # broadcast
        self.subsets_sizes = np.reshape(subsets_sizes, (self.subset_num, 1))

        # tile to (subset_num, label_type_num)
        subsets_sizes = np.tile(self.subsets_sizes, (1, label_type_num))
        # print("shape of subsets_sizes: ", subsets_sizes.shape)
        probs = np.zeros(shape=(self.subset_num, label_type_num), dtype=float)
        # get data sample num from dirichlet distrubution
        alphas = []
        for i in range(self.subset_num):
            if self.alpha_range[0] == self.alpha_range[1]:
                alpha = self.alpha_range[0]
            else:
                alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            alphas.append(alpha)
            # print("alpha: ", alpha)
            alpha_list = [alpha] * label_type_num

            probs[i] = np.random.dirichlet(alpha_list)
        # print("alphas: ", alphas[:5])
        # print("probs: ", probs[:15])
        # broadcast
        distributions: np.ndarray = np.multiply(subsets_sizes, probs)
        distributions.round()
        distributions = distributions.astype(np.int32)

        # print("distributions: ", distributions[:5])

        self.distributions = distributions
        return distributions
    
    def get_subsets(self) -> 'list[Subset]':
        if self.distributions is None:
            self.get_distributions()

        categorized_indexes = self.dataset_categorize(self.dataset)
        self.subsets = []
        # print("distributions: ", self.distributions[:5])
        # print("categorized_indexes: ", categorized_indexes[:5])
        for distribution in self.distributions:
            subset_indexes = []
            for i, num in enumerate(distribution):
                subset_indexes.extend(categorized_indexes[i][:num])
                categorized_indexes[i] = categorized_indexes[i][num:]
            self.subsets.append(Subset(self.dataset, subset_indexes))

        return self.subsets


    # def check_distribution(self, num: int) -> np.ndarray:
    #     subsets = self.subsets[:num]
    #     distributions = np.zeros((num, self.label_type_num), dtype=np.int)
    #     targets = self.dataset.targets

    #     for i, subset in enumerate(subsets):
    #         for j, index in enumerate(subset.indices):
    #             category = targets[index]
    #             distributions[i][category] += 1

    #     return distributions


    # def draw(self, num: int=None, filename: str="./pic/distribution.png"):
    #     if self.distributions is None:
    #         self.get_distributions()
    #     if num is None:
    #         num = len(self.distributions)

    #     xaxis = np.arange(num)
    #     base = np.zeros(shape=(num,))
    #     for i in range(self.distributions.shape[1]):
    #         plt.bar(xaxis, self.distributions[:,i][0:num], bottom=base)
    #         base += self.distributions[:,i][0:num]

    #     plt.rc('font', size=16)
    #     plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

    #     plt.xlabel('Clients', fontsize=20)
    #     plt.ylabel('Distribution', fontsize=20)
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     # plt.grid(True)
    #     # plt.legend()

    #     # plt.savefig('no_selection.pdf')
    #     plt.savefig(filename)
    #     plt.clf()



class DatasetReader:
    class DatasetName(enum.Enum):
        MNIST = 0
        FashionMNIST = 1
        CIFAR10 = 2
        CIFAR100 = 3
        SpeechCommands = 4
    def __init__(self, dataset_name, path: str):
        self.dataset_name = dataset_name
        pass

    def read(self):
        pass

    @staticmethod
    def load_dataset_CIFAR(data_path: str, dataset_type: str="both"):
        # enhance
        # Use the torch.transforms, a package on PIL Image.
        transform_enhanc_func = tvtf.Compose([
            tvtf.RandomHorizontalFlip(p=0.5),
            tvtf.RandomCrop(32, padding=4, padding_mode='edge'),
            tvtf.ToTensor(),
            tvtf.Lambda(lambda x: x.mul(255)),
            tvtf.Normalize([125., 123., 114.], [1., 1., 1.])
            ])

        # transform
        transform_func = tvtf.Compose([
            tvtf.ToTensor(),
            tvtf.Lambda(lambda x: x.mul(255)),
            tvtf.Normalize([125., 123., 114.], [1., 1., 1.])
            ])

        trainset, testset = None, None
        if dataset_type != "test":
            trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                download=True, transform=transform_enhanc_func)
        if dataset_type != "train":
            testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                download=True, transform=transform_func)

        return (trainset, testset)


