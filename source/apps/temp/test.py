
import time
from source.tasks.sc import SCAggregatorTask, SCTaskHelper, SCTrainerTask, SCDatasetPartitionerByUser

from torch.multiprocessing import Process, Pipe


# trainset, testset = SCTaskHelper.get_datasets("./dataset/raw/")
# p = SCDatasetPartitionerByUser(trainset, None, None, None)

# subsets = p.get_pfl_subsets(100, 0.3)

# SCDatasetPartitionerByUser.save_subsets(subsets[:10], "./dataset/partitioned/sc100/")

subsets = SCDatasetPartitionerByUser.load_subsets(5, "./dataset/partitioned/sc100/")

def spawn_task(trainset, testset,):
    task = SCTrainerTask(trainset, testset, 2, 0.01, 10, "cuda")
    for i in range(10):
        task.train()
        accu, loss = task.test()
        print(accu, loss)

ps = []
for i in range(1):
    p = Process(target=spawn_task, args=(subsets[i][0], subsets[i][1]))
    p.start()
    ps.append(p)

