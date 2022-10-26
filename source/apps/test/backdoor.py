

import numpy as np

from torch import nn

from source.tasks.sc import SCTaskHelper, SCTrainerTask

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
distribution = SCTaskHelper.get_label_distri_by_speaker(trainset)
distribution = np.array([ v for k, v in distribution.items() ], dtype=np.int32)

print(distribution.sum(0))
np.sort(distribution, axis=1)
np.savetxt("./exp_data/distribution_all.csv", distribution, delimiter=",", fmt="%d")
# rows = np.argwhere(np.sum(distribution, axis=1) >= 100)
# print(len(rows))
# rows = np.argwhere(np.sum(distribution, axis=1) >= 150)
# print(len(rows))
# rows = np.argwhere(np.sum(distribution, axis=1) >= 200)
# print(len(rows))

# print(distribution[rows])
distribution = [ dist for dist in distribution if np.sum(dist) >= 100 ]

np.savetxt("./exp_data/distribution_data100.csv", distribution, delimiter=",", fmt="%d")


# print("Loading model...")
# task = SCTrainerTask(trainset, testset, 1, 0.01, 256, "cuda")

# for i in range(10):
#     print("Training model...")
#     task.train()
#     print("Testing model...")
#     accu, loss = task.test()
#     print(f"Accuracy: {accu}, Loss: {loss}")
