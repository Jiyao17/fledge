

import json
import enum

import torchvision
from torch.utils.data import Dataset, Subset
import torchvision.transforms as tvtf

class DatasetPartitioner:

    class Config:
        def __init__(self, dataset: str, num_partitions: int):
            self.dataset = dataset
            self.num_partitions = num_partitions

    def __init__(self, config: Config):
        self.config = config

    def partition(self):
        pass

    def save(self, path: str):
        json.dump(self.config, open(path, 'w'))
        pass

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

