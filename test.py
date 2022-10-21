
import time
from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser

from torch.multiprocessing import Queue, Process, Pipe


def generate_subsets():
    trainset, testset = SCTaskHelper.get_raw_datasets("./dataset/raw/")
    p = SCDatasetPartitionerByUser(trainset, None, None, None)

    subsets = p.get_pfl_subsets(100, 0.3)

    SCDatasetPartitionerByUser.save_subsets(subsets[:10], "./dataset/partitioned/sc100/")

def train_proc(trainset, testset):
    task = SCTrainerTask(trainset, testset, 3, 0.01, 10, "cuda")
    for i in range(100):
        task.train()
        accu, loss = task.test()
        print(accu, loss)

def load_and_test_parallel():

    subsets = SCDatasetPartitionerByUser.load_subsets(5, "./dataset/partitioned/sc100/")
    for i in range(5):
        p = Process(target=train_proc, args=(subsets[i][0], subsets[i][1]))
        p.start()


def load_and_test_sequential():
    subsets = SCDatasetPartitionerByUser.load_subsets(5, "./dataset/partitioned/sc100/")
    tasks: list[SCTrainerTask] = []
    for i in range(5):
        task = SCTrainerTask(subsets[i][0], subsets[i][1], 3, 0.01, 10, "cuda")
        tasks.append(task)

    for i in range(100):
        for task in tasks:
            task.train()
            accu, loss = task.test()
            print(accu, loss)
# generate_subsets()
# load_and_test_sequential()
load_and_test_parallel()


