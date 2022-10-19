

import numpy as np

from torch import nn

from source.sc import SCTaskHelper, SCTrainerTask

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def flatten_neural_network(model: nn.Module):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])

print("Loading datasets...")
trainset, testset = SCTaskHelper.get_datasets("./dataset/raw")
print("Loading model...")
task = SCTrainerTask(trainset, testset, 1, 0.01, 256, "cuda")
print("Training model...")
task.train()
print("Testing model...")
task.test()
