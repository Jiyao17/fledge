

import numpy as np

import torch
from torch import nn


def vec_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def vec_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def flatten_model(model: nn.Module):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def flatten_model_sd(model_sd: dict[str, torch.Tensor]):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.cpu().flatten() for p in model_sd.values()])

def model_cosine_similarity(m1, m2):
    """
    Calculate the cosine similarity between two models.
    """
    v1 = flatten_model(m1)
    v2 = flatten_model(m2)
    return vec_cosine_similarity(v1, v2)

def model_sd_cosine_similarity(m1_sd, m2_sd):
    """
    Calculate the cosine similarity between two models.
    """
    v1 = flatten_model_sd(m1_sd)
    v2 = flatten_model_sd(m2_sd)
    return vec_cosine_similarity(v1, v2)

def grads_cosine_similarity(global_model: nn.Module, local_models: list[nn.Module], safe_mode=True):
    """
    Calculate the cosine similarity between two models.
    """
    global_sd = global_model.state_dict()
    local_sds = [m.state_dict() for m in local_models]
    if safe_mode:
        global_sd = {k: v.detach().cpu() for k, v in global_sd.items()}
        local_sds = [{k: v.detach().cpu() for k, v in sd.items()} for sd in local_sds]
    global_vec = flatten_model_sd(global_sd)
    local_vecs = [flatten_model_sd(sd) for sd in local_sds]

    grads = [local_vec - global_vec for local_vec in local_vecs]
    global_grad = np.mean(grads, axis=0)

    cosine_diffs = [vec_cosine_similarity(global_grad, grad) for grad in grads]

    return cosine_diffs