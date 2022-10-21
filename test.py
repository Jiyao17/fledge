
import time
from source.dataset import RealSubset
from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser

from torch.multiprocessing import Queue, Process, Pipe

import torch
import numpy as np



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

    trainset = subsets[0][0]
    testset = subsets[0][1]


def generate_my_dataset():
    X = range(100)
    X = np.array(X, dtype=np.float32)
    Y = [ i ** 2 + i + 1 for i in X]
    Y = [ y + np.random.normal(0, 0.1) for y in Y]
    Y = np.array(Y, dtype=np.float32)

    dataset = [ (x, y) for x, y in zip(X, Y)]

    return dataset

def test_process(dataset):
    # ds = generate_my_dataset()
    # trainset = RealSubset(dataset, 0, 80)
    # testset = RealSubset(dataset, 80, 100)

    for data in dataset:
        x, y = data
        print(x, y)

    time.sleep(10)

# ds = generate_my_dataset()

# rss = RealSubset(ds, range(len(ds)))

# for i in range(5):
#     p = Process(target=test_process, args=(rss,))
#     p.start()


    # for i in range(5):
        # task = SCTrainerTask(subsets[i][0], subsets[i][1], 3, 0.01, 10, "cuda")
        # tasks.append(task)

    # for i in range(100):
    #     for task in tasks:
    #         task.train()
    #         accu, loss = task.test()
    #         print(accu, loss)
# generate_subsets()
# load_and_test_sequential()
# load_and_test_parallel()


