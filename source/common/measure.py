

import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt


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
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model_sd.values()])
    
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

def model_euclidean_distance(m1, m2):
    """
    Calculate the euclidean distance between two models.
    """
    v1 = flatten_model(m1)
    v2 = flatten_model(m2)
    return vec_euclidean_distance(v1, v2)

def model_sd_euclidean_distance(m1_sd, m2_sd):
    """
    Calculate the euclidean distance between two models.
    """
    v1 = flatten_model_sd(m1_sd)
    v2 = flatten_model_sd(m2_sd)
    return vec_euclidean_distance(v1, v2)

def get_updates(global_model: nn.Module, local_models: 'list[nn.Module]'):
    """
    get grads of all local models
    """
    global_sd: dict[str, torch.Tensor] = global_model.state_dict()
    local_sds: 'list[dict[str, torch.Tensor]]' = [m.state_dict() for m in local_models]
    global_vec = flatten_model_sd(global_sd)
    local_vecs = [flatten_model_sd(sd) for sd in local_sds]

    grads = [local_vec - global_vec for local_vec in local_vecs]

    return grads

def cosine_deviation(vecs) -> np.ndarray:
    """
    Calculate the cosine similarity between the average vec and each vec.
    """
    avg_vec = np.mean(vecs, axis=0)
    cosine_deviation = [vec_cosine_dis(avg_vec, vec) for vec in vecs]

    return np.array(cosine_deviation)

def cosine_diff_matrix(vecs) -> np.ndarray:
    """
    Calculate the cosine similarity between two vecs.
    """
    cosine_diffs = np.zeros((len(vecs), len(vecs)))
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            cosine_diffs[i, j] = vec_cosine_dis(vecs[i], vecs[j])

    return cosine_diffs

def euclidean_deviation(vecs) -> np.ndarray:
    """
    Calculate the euclidean distance between the average vec and each vec.
    """
    avg_vec = np.mean(vecs, axis=0)
    euclidean_deviation = [vec_euclidean_distance(avg_vec, vec) for vec in vecs]

    return np.array(euclidean_deviation)

def euclidean_diff_matrix(vecs) -> np.ndarray:
    """
    Calculate the euclidean distance between two vecs.
    """
    euclidean_diffs = np.zeros((len(vecs), len(vecs)))
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            euclidean_diffs[i, j] = vec_euclidean_distance(vecs[i], vecs[j])

    return euclidean_diffs

def grads_cosine_deviation(global_model: nn.Module, local_models: 'list[nn.Module]'):
    """
    Calculate the cosine similarity between global average grad and local grads.
    """
    grads = get_updates(global_model, local_models)
    devi = cosine_deviation(grads)

    return devi

def grads_cosine_diff(global_model: nn.Module, local_models: 'list[nn.Module]'):
    """
    Calculate the cosine similarity between two models.
    """
    grads = get_updates(global_model, local_models)
    diffs = cosine_diff_matrix(grads)

    return diffs

def grads_euclidean_deviation(global_model: nn.Module, local_models: 'list[nn.Module]'):
    """
    Calculate the euclidean distance between global average grad and local grads.
    """
    grads = get_updates(global_model, local_models)
    devi = euclidean_deviation(grads)

    return devi

def grads_euclidean_diff(global_model: nn.Module, local_models: 'list[nn.Module]'):
    """
    Calculate the euclidean distance between two models.
    """
    grads = get_updates(global_model, local_models)
    diffs = euclidean_diff_matrix(grads)

    return diffs

def plot_diff_by_client(diffs: 'np.ndarray', result_file):
    """
    cosine distance between global model and clients' gradients
    one picture for one iteration
    one dot for each other client
    one x for each client
    """
    if len(diffs) == 0:
        return


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:len(diffs)]
    dot_colors = []
    labels = []
    plt.figure()
    for j in range(diffs.shape[0]): # each client
        x = []
        y = []
        for k in range(diffs.shape[1]): # each other client
            if j != k:
                x.append(k)
                y.append(diffs[j][k])
                dot_colors.append(colors[j])
    
        plt.scatter(x, y, label=f'client {j}')
    plt.legend()
    plt.savefig(result_file)
    plt.close()

def plot_devi_by_client(devi: 'np.ndarray', result_file):
    """
    cosine distance between global model and clients' gradients
    one picture for each epoch
    one dot for each clientd
    """
    plt.figure()
    plt.scatter(range(len(devi)), devi)
    plt.legend()
    plt.savefig(result_file)
    plt.close()

