
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

def train_proc(id):
    trainset = SCDatasetPartitionerByUser.load_subset(
        "./dataset/partitioned/sc100/subtrainset_{}.pt".format(id))
    testset = SCDatasetPartitionerByUser.load_subset(
        "./dataset/partitioned/sc100/subtestset_{}.pt".format(id))

    task = SCTrainerTask(trainset, testset, 3, 0.01, 10, "cuda")
    f = open("test{}.txt".format(id), "a")
    for i in range(100):
        task.train()
        accu, loss = task.test()
        f.write("{} {}\n".format(accu, loss))
        f.flush()

def load_and_test_parallel():

    for i in range(3):
        p = Process(target=train_proc, args=(i,))
        p.start()

def load_and_test_sequential():
    pass



# generate_subsets()
# load_and_test_sequential()
load_and_test_parallel()
# train_proc(0)


