

from .source.tasks.sc import SCTrainerTask, SCTaskHelper
import torch.nn.functional as F

from torchaudio.transforms import Resample


class SCTrainerTask():

    def __init__(self, epochs: int, lr: float, batch_size: int, device: str):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

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
        return SCTaskHelper.test_model(self.model, self.test_dataloader, self.device)


