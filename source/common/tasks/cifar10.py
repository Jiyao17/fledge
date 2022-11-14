
from torch.utils.data import Dataset
from ..data import DatasetPartitionerDirichlet


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
import torch.nn.functional as F
import torchvision.transforms as tvtf
import torchvision

from ..task import Task, AggregatorTask, TaskHelper


# Code for CIFAR ResNet is modified from https://github.com/itchencheng/pytorch-residual-networks


class ResBlock(nn.Module):
    def __init__(self, in_chann, chann, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        
        if (x.shape == y.shape):
            z = x
        else:
            z = F.avg_pool2d(x, kernel_size=2, stride=2)            

            x_channel = x.size(1)
            y_channel = y.size(1)
            ch_res = (y_channel - x_channel)//2

            pad = (0, 0, 0, 0, ch_res, ch_res)
            z = F.pad(z, pad=pad, mode="constant", value=0)

        z = z + y
        z = F.relu(z)
        return z


class BaseNet(nn.Module):
    
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.convs  = self._make_layers(n)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        
        x = F.relu(x)
        
        x = self.convs(x)
        
        x = self.avgpool(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 16
        chann = 16
        stride = 1
        for i in range(3):
            for j in range(n):
                if ((i > 0) and (j == 0)):
                    in_chann = chann
                    chann = chann * 2
                    stride = 2

                layers += [self.Block(in_chann, chann, stride)]

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)


class CIFAR10ResNet(BaseNet):
    def __init__(self, n=3):
        super().__init__(ResBlock, n)



class CIFAR10TrainerTask(Task):
    def __init__(self, trainset: Dataset, testset: Dataset, epochs: int, lr: float, batch_size: int, device: str):
        super().__init__(trainset, testset, epochs, lr, batch_size, device)
        
        self.model = CIFAR10ResNet()
        self.loss_fn = CIFAR10TaskHelper.loss_fn
        self.trainloader = DataLoader(self.trainset, batch_size, False) #, drop_last=True
        self.testloader = DataLoader(self.testset, batch_size, False) #, drop_last=True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01) #

    def update(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            for samples, labels in self.trainloader:
                pred = self.model(samples.to(self.device))
                loss = self.loss_fn(pred, labels.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def test(self):
        return CIFAR10TaskHelper.test_model(self.model, self.testloader, self.device, self.loss_fn)


class CIFAR10AggregatorTask(AggregatorTask):
    def __init__(self, trainset: Dataset, testset: Dataset, 
        epochs: int, lr: float, batch_size: int, 
        device: str='cpu',
        ):
        super().__init__(trainset, testset, epochs, lr, batch_size, device)
        
        self.model = CIFAR10ResNet().to(device)
        self.loss_fn = CIFAR10TaskHelper.loss_fn
        self.testloader = DataLoader(self.testset, batch_size, False) #, drop_last=True

    def test(self):
        return CIFAR10TaskHelper.test_model(self.model, self.testloader, self.device, self.loss_fn)

class CIFAR10TaskHelper(TaskHelper):
    
    loss_fn = F.cross_entropy
    AggregatorTaskClass: AggregatorTask = CIFAR10AggregatorTask
    TrainerTaskClass: Task = CIFAR10TrainerTask

    def __init__(self, dataset: Dataset):
        self.dataset = dataset


    @staticmethod
    def get_datasets(path) -> 'tuple[Dataset, Dataset]':
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
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_enhanc_func)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_func)

        return (trainset, testset)

    @staticmethod
    def test_model(model: nn.Module, testloader: DataLoader, device: str='cuda', loss_fn=F.cross_entropy) \
        -> 'tuple[float, float]':
        model.to(device)
        model.eval()

        loss = loss_fn
        size = 0
        correct: float = 0.0
        test_loss: float = 0.0

        # with torch.no_grad():
        for samples, labels in testloader:
            pred = model(samples.to(device))
            correct += (pred.argmax(1) == labels.to(device)).type(torch.float).sum().item()
            test_loss += loss(pred, labels.to(device)).item()

            size += len(samples)

        correct /= 1.0*size
        test_loss /= 1.0*len(testloader)
        return correct, test_loss

    def get_label_types(self) -> 'list[int]':
        return list(range(10))

    def get_targets(self) -> list:
        return self.dataset.targets



class CIFAR10PartitionerDrichlet(CIFAR10TaskHelper, DatasetPartitionerDirichlet):
    def __init__(self, dataset: Dataset, subset_num: int = 100, data_num_range: 'tuple[int]' = ..., alpha_range: 'tuple[float, float]' = ...):
        CIFAR10TaskHelper.__init__(self, dataset)
        DatasetPartitionerDirichlet.__init__(self, dataset, 
            subset_num, data_num_range, alpha_range)


