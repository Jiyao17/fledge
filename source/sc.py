

# Speech Commands Dataset supporting class


import os
from typing import overload
from importlib_metadata import distribution

import torch
from torch import nn
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import DataLoader

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
from torch import nn, optim, Tensor

from .task import TrainerTask
from .dataset import DatasetPartitioner

import numpy as np


class SCModel(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class SCTaskHelper:
    labels: list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, dataset_type, data_path):
            super().__init__(root=data_path, download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.join(self._path, line.strip()) for line in fileobj]

            if dataset_type == "validation":
                self._walker = load_list("validation_list.txt")
            elif dataset_type == "testing":
                self._walker = load_list("testing_list.txt")
            elif dataset_type == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

    @staticmethod
    def get_label_distri_by_speaker(dataset: Dataset) -> 'dict[str, list]':
        """
        Analyze the dataset.
        return: dict{'speaker_id': [labels_nums]}
        """
        label_num = len(SCTaskHelper.labels)
        distribution: dict[str:list] = {}
        for i in range(len(dataset)):
            waveform, sample_rate, label, speaker_id, utterance_number = dataset[i]
            if speaker_id not in distribution.keys():
                distribution[speaker_id] = np.zeros(label_num, dtype=np.int32)
            label_index = SCTaskHelper.labels.index(label)
            distribution[speaker_id][label_index] += 1

        # dist_arr = np.array([ v for k, v in distribution.items() ], dtype=np.int32)
        return distribution

    @staticmethod
    def get_index_distri_by_speaker(dataset: Dataset) -> 'dict[str, list]':
        """
        Analyze the dataset.
        return: dict{'speaker_id': [data_indexs]}
        """
        distr_by_speaker: dict[str, list] = {}

        for i in range(len(dataset)):
            wave_form, sample_rate, label, speaker_id, utterance_number = dataset[i]
            if speaker_id not in distr_by_speaker.keys():
                distr_by_speaker[speaker_id] = []
            distr_by_speaker[speaker_id].append(i)

        return distr_by_speaker


    @staticmethod
    def get_datasets(data_path: str) -> 'tuple[Dataset, Dataset]':
        testset = SCTaskHelper.SubsetSC("testing", data_path)
        trainset = SCTaskHelper.SubsetSC("training", data_path)

        removed_train = []
        removed_test = []
        for i in range(len(trainset)):
            waveform, sample_rate, label, speaker_id, utterance_number = trainset[i]
            if waveform.shape[-1] != 16000:
                removed_train.append(i)

        trainset = Subset(trainset, list(set(range(len(trainset))) - set(removed_train)))

        for i in range(len(testset)):
            waveform, sample_rate, label, speaker_id, utterance_number = testset[i]
            if waveform.shape[-1] != 16000:
                removed_test.append(i)
                # testset._walker.remove(testset._walker[i])
                # removed_test += 1
        
        testset = Subset(testset, list(set(range(len(testset))) - set(removed_test)))

        print("Data number removed from trainset: ", len(removed_train))
        print("Data number removed from testset: ", len(removed_test))
        return (trainset, testset)

    @staticmethod
    def get_dataloader(loader_type: str, dataset=None, device=None, batch_size=10):
        """
        loader_type: train or test
        """
        if device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        if loader_type != "train":
            dataloader = DataLoader(
                dataset,
                batch_size=500,
                shuffle=False,
                drop_last=True,
                collate_fn=SCTaskHelper.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=SCTaskHelper.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
            )
            
        
        return dataloader

    @staticmethod
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(SCTaskHelper.labels.index(word))

    @staticmethod
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return SCTaskHelper.labels[index]

    @staticmethod    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    @staticmethod
    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [SCTaskHelper.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = SCTaskHelper.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @staticmethod
    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    @staticmethod
    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    @staticmethod
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SCTrainerTask(TrainerTask):

    def __init__(self, trainset: Dataset, testset: Dataset, epochs: int, lr: float, batch_size: int, device: str):
        super().__init__(trainset, testset, epochs, lr, batch_size, device)

        self.loss_fn = F.nll_loss

        # print(len(self.trainset))
        # if self.testset is not None:
        #     waveform, sample_rate, label, speaker_id, utterance_number = self.testset[0]
        # else:
        #     waveform, sample_rate, label, speaker_id, utterance_number = self.trainset[0]
        # new_sample_rate = 8000

        transform = Resample(orig_freq=16000, new_freq=8000, )
        # transformed: Resample = transform(waveform)
        self.transform = transform.to(device)
        # waveform = waveform.to(device)
        # tranformed = self.transform(waveform).to(device)

        self.model = SCModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        # step_size = self.config.lr_interval * self.config.group_epoch_num * self.config.local_epoch_num
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

        if trainset is not None:
            self.train_dataloader = SCTaskHelper.get_dataloader("train", trainset, device, batch_size)
        if testset is not None:
            self.test_dataloader = SCTaskHelper.get_dataloader("test", testset, device, batch_size)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        self.transform = self.transform.to(self.device)

        for epoch in range(self.epochs):
            for data, target in self.train_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                # apply transform and model on whole batch directly on device
                data = self.transform(data)
                output = self.model(data)
                # negative log-likelihood for a tensor of size (batch x 1 x n_output)
                loss = self.loss_fn(output.squeeze(), target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        dataset_size = len(self.test_dataloader.dataset)
        correct, loss = 0, 0
        for data, target in self.test_dataloader:
            data = data.to(self.device)
            target = target.to(self.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output: Tensor = self.model(data)

            pred = SCTaskHelper.get_likely_index(output)
            loss += self.loss_fn(output.squeeze(), target).item()

            # pred = output.argmax(dim=-1)
            correct += SCTaskHelper.number_of_correct(pred, target)

        correct /= 1.0*dataset_size
        loss /= 1.0*dataset_size

        return correct, loss


class SCDatasetPartitioner(DatasetPartitioner):

    def __init__(self, dataset: Dataset, subset_num: int, data_num_range: tuple, alpha_range: tuple):
        super().__init__(dataset, subset_num, data_num_range, alpha_range)

    def get_label_types(self) -> 'list[str]':
        return SCTaskHelper.labels

    def dataset_categorize(self) -> 'list[list[int]]':
        targets = []
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            targets.append(label)

        categorized_indexes = [[] for i in range(len(self.get_label_types()))]
        for i, target in enumerate(targets):
            categorized_indexes[target].append(i)
        return categorized_indexes


class SCDatasetPartitionerByUser(SCDatasetPartitioner):

    
    def get_subsets(self, data_num_threshold) -> 'list[Dataset]':
        distribution = SCTaskHelper.get_index_distri_by_speaker(self.dataset)
        filtered_distri = {}
        for speaker in distribution.keys():
            if sum(distribution[speaker]) > data_num_threshold:
                filtered_distri[speaker] = distribution[speaker]

        subsets = []
        for speaker in filtered_distri.keys():
            subset = Subset(self.dataset, filtered_distri[speaker])
            subsets.append(subset)
        
        return subsets

