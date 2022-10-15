
from typing import Iterable
import torch
import torchvision
import torchvision.transforms as tvtf
from torch.utils.data.dataset import Dataset, Subset

import matplotlib.pyplot as plt

import numpy as np

import random
from enum import Enum

from utils.model import SpeechCommand


class TaskName(Enum):
    CIFAR = 1
    SPEECHCOMMAND = 2

def quick_draw(values: Iterable, filename: str="./pic/quick_draw.png"):
    plt.plot(values)
    plt.savefig(filename)
    plt.close()

def load_dataset_CIFAR(data_path: str, dataset_type: str):
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

def load_dataset(dataset_name: TaskName, data_path: str="~/projects/fl-grouping/data/", dataset_type: str="both"):
    if dataset_name == TaskName.CIFAR:
        return load_dataset_CIFAR(data_path, dataset_type)
    # elif dataset_name == TaskName.SPEECHCOMMAND:
    #     return load_dataset_SpeechCommand(data_path, dataset_type)


def get_targets_set_as_list(dataset: Dataset) -> list:
    targets = dataset.targets
    if type(targets) is not list:
        targets = targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    return targets_list

def dataset_categorize(dataset: Dataset, task_name:TaskName) -> 'list[list[int]]':
    """
    return value:
    (return list)[i]: list[int] = all indices for category i
    """
    if task_name == TaskName.CIFAR:
        targets = dataset.targets
    elif task_name == TaskName.SPEECHCOMMAND:
        targets = []
        for waveform, sample_rate, label, speaker_id, utterance_number in dataset:
            targets.append(label)
    if type(targets) is not list:
        targets = targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    indices_by_lable = [[] for target in targets_list]
    for i, target in enumerate(targets):
        category = targets_list.index(target)
        indices_by_lable[category].append(i)

    # randomize
    for indices in indices_by_lable:
        random.shuffle(indices)

    # subsets = [Subset(dataset, indices) for indices in indices_by_lable]
    return indices_by_lable


class DatasetPartitioner:
    speech_command_labels: list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']

    @staticmethod
    def plot_distribution(distributions: np.ndarray, num: int, filename: str="./pic/distribution.png"):
        
        xaxis = np.arange(num)
        base = np.zeros(shape=(num,))
        for i in range(distributions.shape[1]):
            plt.bar(xaxis, distributions[:,i][0:num], bottom=base)
            base += distributions[:,i][0:num]

        plt.rc('font', size=16)
        plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

        plt.xlabel('Clients', fontsize=20)
        plt.ylabel('Distribution', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.grid(True)
        # plt.legend()

        # plt.savefig('no_selection.pdf')
        plt.savefig(filename)
        plt.clf()

    def __init__(self, dataset: Dataset, subset_num: int=1000, 
        data_num_range: 'tuple[int]'=(10, 50), 
        alpha_range: 'tuple[float, float]'=(0.05, 0.5), task_name: TaskName=TaskName.CIFAR
        ):
        self.dataset = dataset
        self.subset_num = subset_num
        # range = (min, max)
        self.data_num_range = data_num_range
        self.task_name = task_name
        if task_name == TaskName.CIFAR:

            self.label_type_num = len(get_targets_set_as_list(dataset))
        elif task_name == TaskName.SPEECHCOMMAND:
            self.label_type_num = len(DatasetPartitioner.speech_command_labels)
        # self.alpha = [alpha] * self.label_type_num
        self.alpha_range = alpha_range

        self.distributions: np.ndarray = None
        self.cvs: np.ndarray = None
        self.subsets: list[Dataset] = []
        self.subsets_sizes: np.ndarray = None

    def get_distributions(self):
        subsets_sizes = np.random.randint(self.data_num_range[0], self.data_num_range[1], size=self.subset_num)
        # print("subset_size: ", subsets_sizes[:15])
        # broadcast
        self.subsets_sizes = np.reshape(subsets_sizes, (self.subset_num, 1))

        # tile to (subset_num, label_type_num)
        subsets_sizes = np.tile(self.subsets_sizes, (1, self.label_type_num))
        # print("shape of subsets_sizes: ", subsets_sizes.shape)
        probs = np.zeros(shape=(self.subset_num, self.label_type_num), dtype=float)
        # get data sample num from dirichlet distrubution
        alphas = []
        for i in range(self.subset_num):
            if self.alpha_range[0] == self.alpha_range[1]:
                alpha = self.alpha_range[0]
            else:
                alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            alphas.append(alpha)
            # print("alpha: ", alpha)
            alpha_list = [alpha] * self.label_type_num

            probs[i] = np.random.dirichlet(alpha_list)
        # print("alphas: ", alphas[:5])
        # print("probs: ", probs[:15])
        # broadcast
        distributions: np.ndarray = np.multiply(subsets_sizes, probs)
        distributions.round()
        distributions = distributions.astype(np.int)

        # print("distributions: ", distributions[:5])

        self.distributions = distributions
        return distributions
    
    def get_cvs(self):
        if self.distributions is None:
            self.get_distributions()

        stds = np.std(self.distributions, axis=1)
        self.cvs = stds / np.mean(self.distributions, axis=1)
        return self.cvs

    def get_subsets(self) -> 'list[Subset]':
        if self.distributions is None:
            self.get_distributions()

        categorized_indexes = dataset_categorize(self.dataset, self.task_name)
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

    def check_distribution(self, num: int) -> np.ndarray:
        subsets = self.subsets[:num]
        distributions = np.zeros((num, self.label_type_num), dtype=np.int)
        targets = self.dataset.targets

        for i, subset in enumerate(subsets):
            for j, index in enumerate(subset.indices):
                category = targets[index]
                distributions[i][category] += 1

        return distributions


    def draw(self, num: int=None, filename: str="./pic/distribution.png"):
        if self.distributions is None:
            self.get_distributions()
        if num is None:
            num = len(self.distributions)

        xaxis = np.arange(num)
        base = np.zeros(shape=(num,))
        for i in range(self.distributions.shape[1]):
            plt.bar(xaxis, self.distributions[:,i][0:num], bottom=base)
            base += self.distributions[:,i][0:num]

        plt.rc('font', size=16)
        plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

        plt.xlabel('Clients', fontsize=20)
        plt.ylabel('Distribution', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.grid(True)
        # plt.legend()

        # plt.savefig('no_selection.pdf')
        plt.savefig(filename)
        plt.clf()
