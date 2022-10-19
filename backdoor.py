

import numpy as np

from torch import nn

from utils.dataset import SpeechCommandsPartitioner


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def flatten_neural_network(model: nn.Module):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])



