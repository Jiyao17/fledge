

import numpy as np

from torch import nn

def vec_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def vec_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def flatten_neural_network(model: nn.Module):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])

def model_cosine_similarity(m1, m2):
    """
    Calculate the cosine similarity between two models.
    """
    v1 = flatten_neural_network(m1)
    v2 = flatten_neural_network(m2)
    return vec_cosine_similarity(v1, v2)