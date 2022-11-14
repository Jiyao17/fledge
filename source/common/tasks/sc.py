

# Speech Commands Dataset supporting class


import os

import torch
from torch import nn
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import DataLoader

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
from torch import nn, optim, Tensor

from ..task import Task, AggregatorTask, TaskHelper
from ..data import DatasetPartitioner, DatasetPartitionerDirichlet

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



class SCTrainerTask(Task):

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
            self.test_dataloader = SCTaskHelper.get_dataloader("test", testset, device, len(testset))

    def update(self):
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

    def test(self) -> 'tuple[float, float]':
        accu, loss = SCTaskHelper.test_model(self.model, self.test_dataloader, self.device)
        return accu, loss


class SCAttackerTask(SCTrainerTask):
    def __init__(self, trainset: Dataset, testset: Dataset, epochs: int, lr: float, batch_size: int, device: str):
        super().__init__(trainset, testset, epochs, lr, batch_size, device)

    @staticmethod
    def attacker_dataset_compose(dataset: Dataset, attack_label: str, target_label: str) -> Dataset:
        """
        change all labels to target_type
        """
        pass


class SCAggregatorTask(AggregatorTask):

    def __init__(self, trainset: Dataset=None, testset: Dataset=None, 
        epochs: int=1, lr: float=0.001, batch_size: int=64, 
        device: str="cpu"
        ):
        
        super().__init__(trainset, testset, epochs, lr, batch_size, device)

        self.loss_fn = SCTaskHelper.loss_fn
        self.device = device

        transform = SCTaskHelper.transform
        self.transform = transform.to(device)

        self.model = SCModel().to(device)

        if testset is not None:
            self.test_dataloader = SCTaskHelper.get_dataloader("test", testset, device, 500)

    def test(self):
        return SCTaskHelper.test_model(self.model, self.test_dataloader, self.device)



class SCTaskHelper(TaskHelper):
    """
    Helper class for Speech Commands Task
    Contains common ustils for SC
    
    """

    AggregatorTaskClass: type = SCAggregatorTask
    TrainerTaskClass: type = SCTrainerTask

    labels: 'tuple[str]' = ('backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero')
    loss_fn = F.nll_loss
    transform = Resample(orig_freq=16000, new_freq=8000, )




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
    def attacker_dataset_compose(dataset: Dataset, target_type: str="yes",):
        """
        Compose dataset for attacker
        change all labels to target_type
        """
        for i in range(len(dataset)):
            waveform, sample_rate, label, speaker_id, utterance_number = dataset[i]
            dataset[i] = (waveform, sample_rate, target_type, speaker_id, utterance_number)
        pass

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
    def get_index_distribution_by_speaker(dataset: Dataset) -> 'tuple[dict[str, list]]':
        """
        Analyze the dataset.
        return: dict{'speaker_id': [data_indexs]} and dict{'speaker_id': [labels distribution]}
        """
        index_by_speaker: dict[str, list] = {}
        distribution_by_speaker: dict[str, list] = {}

        for i in range(len(dataset)):
            wave_form, sample_rate, label, speaker_id, utterance_number = dataset[i]
            if speaker_id not in index_by_speaker.keys():
                index_by_speaker[speaker_id] = []
                distribution_by_speaker[speaker_id] = np.zeros(len(SCTaskHelper.labels), dtype=np.int32)
            
            index_by_speaker[speaker_id].append(i)
            label_index = SCTaskHelper.labels.index(label)
            distribution_by_speaker[speaker_id][label_index] += 1

        return index_by_speaker, distribution_by_speaker

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
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=SCTaskHelper.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=SCTaskHelper.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            
        
        return dataloader

    @staticmethod
    def test_model(model: nn.Module, test_dataloader: DataLoader, device: str):
        
        model.to(device)
        transform = SCTaskHelper.transform.to(device)
        model.eval()

        dataset_size = len(test_dataloader.dataset)
        correct, loss = 0, 0
        for data, target in test_dataloader:
            data: Tensor = data.to(device)
            target: Tensor = target.to(device)
            # apply transform and model on whole batch directly on device
            data = transform(data)
            output: Tensor = model(data)

            pred = SCTaskHelper.get_likely_index(output)
            loss += SCTaskHelper.loss_fn(output.squeeze(), target).item()

            # pred = output.argmax(dim=-1)
            correct += SCTaskHelper.number_of_correct(pred, target)

        correct /= 1.0*dataset_size
        loss /= 1.0*dataset_size

        return correct, loss

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
    def number_of_correct(pred: Tensor, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    @staticmethod
    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    @staticmethod
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SCDatasetPartitionHelper:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def get_targets(self) -> list:
        targets = []
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            targets.append(label)
        return targets

    def get_label_types(self) -> 'list[str]':
        return SCTaskHelper.labels


class SCDatasetPartitionerByUser(SCDatasetPartitionHelper, DatasetPartitioner):
    def __init__(self, dataset: Dataset,):
        DatasetPartitioner.__init__(self, dataset)
        SCDatasetPartitionHelper.__init__(self, dataset)


    def get_subsets(self, data_num_threshold) -> 'list[Dataset]':
        # obsolete
        index_by_user, distribution_by_user = SCTaskHelper.get_index_distribution_by_speaker(self.dataset)
        filtered_indices = {}
        filtered_distribution = {}
        for speaker in index_by_user.keys():
            if sum(index_by_user[speaker]) >= data_num_threshold:
                filtered_indices[speaker] = index_by_user[speaker]
                filtered_distribution[speaker] = distribution_by_user[speaker]

        subsets = []
        for speaker in filtered_indices.keys():
            subset = Subset(self.dataset, filtered_indices[speaker])
            subsets.append(subset)
        
        self.distributions_by_user = filtered_distribution
        self.distributions = filtered_distribution

        return subsets

    def get_pfl_subsets(self, subset_num: int, 
            data_num_threshold: int, test_frac: float) \
            -> 'list[tuple[Dataset, Dataset]]':
        """
        generate subsets for pfl
        trainset and testset for each user
        """
        indices_by_user, distributions_by_user = SCTaskHelper.get_index_distribution_by_speaker(self.dataset)
        
        user_subsets = []
        self.distributions = np.zeros((subset_num, len(SCTaskHelper.labels)))
        cnt = 0
        for speaker in indices_by_user.keys():
            if len(indices_by_user[speaker]) >= data_num_threshold:
                if cnt >= subset_num:
                    break
                # randomize user data
                np.random.shuffle(indices_by_user[speaker])
                # split user data into train and test
                test_num = int(len(indices_by_user[speaker]) * test_frac)
                trainset = Subset(self.dataset, indices_by_user[speaker][test_num:])
                testset = Subset(self.dataset, indices_by_user[speaker][:test_num])

                user_subsets.append((trainset, testset))

                # record distribution
                self.distributions[cnt] = distributions_by_user[speaker]
                cnt += 1
        
        if cnt < subset_num:
            raise Exception('Not enough qualified users')
        
        return user_subsets


class SCDatasetPartitionerDirichlet(SCDatasetPartitionHelper, DatasetPartitionerDirichlet):
    def __init__(self, dataset: Dataset, 
            subset_num: int, data_num_range, alpha_range
            ):
        SCDatasetPartitionHelper.__init__(self, dataset)
        DatasetPartitionerDirichlet.__init__(self, dataset, 
            subset_num, data_num_range, alpha_range)

