
from torch import nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
from torch import nn, optim, Tensor

from ..source.task import TrainerTask


class SCTrainerTask(TrainerTask):
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset, epochs: int, lr: float, batch_size: int, device: str):
        super().__init__(model, trainset, testset, epochs, lr, batch_size, device)

        self.loss_fn = F.nll_loss

        # print(len(self.trainset))
        if self.testset is not None:
            waveform, sample_rate, label, speaker_id, utterance_number = self.testset[0]
        else:
            waveform, sample_rate, label, speaker_id, utterance_number = self.trainset[0]
        new_sample_rate = 8000
        transform = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        # transformed: Resample = transform(waveform)
        self.transform = transform.to(device)
        waveform = waveform.to(device)
        self.tranformed = self.transform(waveform).to(device)
        self.model = SpeechCommand(n_input=self.tranformed.shape[0], n_output=len(self.labels)).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.0001)
        step_size = self.config.lr_interval * self.config.group_epoch_num * self.config.local_epoch_num
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

        if trainset is not None:
            self.train_dataloader = TaskSpeechCommand.get_dataloader("train", self.config, trainset)
        if testset is not None:
            self.test_dataloader = TaskSpeechCommand.get_dataloader("test", self.config, None, testset)

    def train(self):
        pass

class TaskSpeechCommand():

    labels: list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset, data_path):
            super().__init__(root=data_path, download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.join(self._path, line.strip()) for line in fileobj]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

    def __init__(self, config: Config, trainset=None, testset=None):
        self.config = copy.deepcopy(config)
        self.trainset = trainset
        self.testset = testset

        self.scheduler = None
        self.loss_fn = F.nll_loss

        # print(len(self.trainset))
        if self.testset is not None:
            waveform, sample_rate, label, speaker_id, utterance_number = self.testset[0]
        else:
            waveform, sample_rate, label, speaker_id, utterance_number = self.trainset[0]
        new_sample_rate = 8000
        transform = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        # transformed: Resample = transform(waveform)
        self.transform = transform.to(self.config.device)
        waveform = waveform.to(self.config.device)
        self.tranformed = self.transform(waveform).to(self.config.device)
        self.model = SpeechCommand(n_input=self.tranformed.shape[0], n_output=len(self.labels)).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.0001)
        step_size = self.config.lr_interval * self.config.group_epoch_num * self.config.local_epoch_num
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

        if trainset is not None:
            self.train_dataloader = TaskSpeechCommand.get_dataloader("train", self.config, trainset)
        if testset is not None:
            self.test_dataloader = TaskSpeechCommand.get_dataloader("test", self.config, None, testset)

    @staticmethod
    def get_datasets(config: Config) -> 'tuple[Dataset, Dataset]':
        testset = TaskSpeechCommand.SubsetSC("testing", config.data_path)
        trainset = TaskSpeechCommand.SubsetSC("training", config.data_path)

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
    def get_dataloader(loader, config: Config, trainset=None, testset=None, ):

        if config.device == torch.device("cuda"):
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # test dataloader
        if loader != "train":
            test_dataloader = DataLoader(
                    testset,
                    batch_size=1000,
                    shuffle=False,
                    drop_last=True,
                    collate_fn=TaskSpeechCommand.collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    )
        if loader != "test":
        # train dataloader
        # if 0 <= self.config.reside and self.configs.reside <= self.configs.client_num:
        #     data_num = self.configs.l_data_num
        #     reside = self.configs.reside
        #     self.trainset = Subset(Task.trainset,
        #         Task.trainset_perm[data_num*reside: data_num*(reside+1)])
            # self.trainset = trainset
            train_dataloader = DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=TaskSpeechCommand.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
            )
            
        if loader == "train":
            return train_dataloader
        elif loader == "test":
            return test_dataloader
        else:
            return train_dataloader, test_dataloader

        

    def train(self):
        self.model.to(self.config.device)
        self.model.train()
        self.transform = self.transform.to(self.config.device)
        for data, target in self.train_dataloader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = self.loss_fn(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

    def test(self, test_dataloader):
        self.model.to(self.config.device)
        self.model.eval()

        dataset_size = len(test_dataloader.dataset)
        correct, loss = 0, 0
        for data, target in test_dataloader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = TaskSpeechCommand.get_likely_index(output)
            loss += self.loss_fn(output.squeeze(), target).item()

            # pred = output.argmax(dim=-1)
            correct += TaskSpeechCommand.number_of_correct(pred, target)

        correct /= 1.0*dataset_size
        loss /= 1.0*dataset_size

        return correct, loss

    @staticmethod
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(TaskSpeechCommand.labels.index(word))

    @staticmethod
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return TaskSpeechCommand.labels[index]

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
            targets += [TaskSpeechCommand.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = TaskSpeechCommand.pad_sequence(tensors)
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
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
