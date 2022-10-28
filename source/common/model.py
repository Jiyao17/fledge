

import numpy as np

import torch
from torch import nn


def vec_cosine_dis(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def vec_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def flatten_model(model: nn.Module):
    """
    Flatten the neural network model into a vector.
    """
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def flatten_model_sd(model_sd: 'dict[str, torch.Tensor]'):
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
    return vec_cosine_dis(v1, v2)

def model_sd_cosine_similarity(m1_sd, m2_sd):
    """
    Calculate the cosine similarity between two models.
    """
    v1 = flatten_model_sd(m1_sd)
    v2 = flatten_model_sd(m2_sd)
    return vec_cosine_dis(v1, v2)

def get_grads(global_model: nn.Module, local_models: 'list[nn.Module]', safe_mode=True):
    """
    get grads of all local models
    """
    global_sd: dict[str, torch.Tensor] = global_model.state_dict()
    local_sds: 'list[dict[str, torch.Tensor]]' = [m.state_dict() for m in local_models]
    if safe_mode:
        global_sd = {k: v.detach().cpu() for k, v in global_sd.items()}
        local_sds = [{k: v.detach().cpu() for k, v in sd.items()} for sd in local_sds]
    global_vec = flatten_model_sd(global_sd)
    local_vecs = [flatten_model_sd(sd) for sd in local_sds]

    grads = [local_vec - global_vec for local_vec in local_vecs]

    return grads

def grads_cosine_deviation(global_model: nn.Module, local_models: 'list[nn.Module]', safe_mode=True):
    """
    Calculate the cosine similarity between two models.
    """
    grads = get_grads(global_model, local_models, safe_mode=safe_mode)
    global_grad = np.mean(grads, axis=0)

    cosine_deviation = [vec_cosine_dis(global_grad, grad) for grad in grads]

    return cosine_deviation

def grads_cosine_diff(global_model: nn.Module, local_models: 'list[nn.Module]', safe_mode=True):
    """
    Calculate the cosine similarity between two models.
    """
    grads = get_grads(global_model, local_models, safe_mode=safe_mode)
    global_grad = np.mean(grads, axis=0)


    cosine_diffs = np.zeros((len(grads), len(grads)))
    for i in range(len(grads)):
        for j in range(len(grads)):
            cosine_diffs[i, j] = vec_cosine_dis(grads[i], grads[j])

    return cosine_diffs


