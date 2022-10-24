
import time
import json

import sys
sys.path.append('..')
from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser

from torch.multiprocessing import Process, Pipe


trainset, testset = SCTaskHelper.get_datasets("../dataset/raw/")
p = SCDatasetPartitionerByUser(trainset, None, None, None)

subsets = p.get_pfl_subsets(100, 0.3)
folder = "../dataset/partitioned/sc100/"
setting = {
    "min_data_num": 100,
    "test_ratio": 0.3,
}

with open(folder + "setting.json", "w") as f:
    json.dump(setting, f)

for i, subset in enumerate(subsets):
    SCDatasetPartitionerByUser.save_dataset(subset[0], folder+str(i)+".train")
    SCDatasetPartitionerByUser.save_dataset(subset[1], folder+str(i)+".test")

# subsets = SCDatasetPartitionerByUser.load_dataset(5, "./dataset/partitioned/sc100/")

# def spawn_task(trainset, testset,):
#     task = SCTrainerTask(trainset, testset, 2, 0.01, 10, "cuda")
#     for i in range(10):
#         task.train()
#         accu, loss = task.test()
#         print(accu, loss)


