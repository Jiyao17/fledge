

import time

import torch
from torch import multiprocessing as mp

def func(t0, t1):
    print("t0: ", t0)
    print("t1: ", t1)
    t2 = t0 + t1
    time.sleep(100)
    t2 = t2 + 1
    return t2


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    t0 = torch.tensor([1, 2, 3]).cuda().share_memory_()
    t1 = torch.tensor([4, 5, 6]).cuda().share_memory_()

    procs = []
    for i in range(5):
        proc = mp.Process(target=func, args=(t0, t1))
        procs.append(proc)
        proc.start()

    