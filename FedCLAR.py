

# FedCLAR implementation, based on Speech Commands Dataset

from multiprocessing import Process, Pipe
from source.sc import SCTaskHelper, SCTrainerTask
from source.node import Trainer, Aggregator
from source.app import App


if __name__ == "__main__":
    config = App.Config("./dataset/raw")
    pipe_agg, pipe_trainer = Pipe()
    task = SCTrainerTask()
    client = Trainer(task, pipe_trainer)
    aggregator = Aggregator(task, [pipe_agg])