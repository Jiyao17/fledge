
import pickle


import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt


def _read_cifar10_raw(path: str):
    for i in range(1, 6):
        with open(path + f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
            data = pickle.load(f, encoding="bytes")
            if i == 1:
                images = data[b"data"]
                labels = data[b"labels"]
            else:
                images = np.concatenate((images, data[b"data"]))
                labels = np.concatenate((labels, data[b"labels"]))

    with open(path + "cifar-10-batches-py/test_batch", "rb") as f:
        data = pickle.load(f, encoding="bytes")
        test_images = data[b"data"]
        test_labels = data[b"labels"]

    return images, labels, test_images, test_labels

def read_cifar10(path: str):
    """
    read cifar10 as trainset and testset
    dataset format: (img: numpy.ndarray, label)
    """
    images, labels, test_images, test_labels = _read_cifar10_raw(path)
    trainset = []
    testset = []
    for i in range(len(images)):
        trainset.append((images[i], labels[i]))
    for i in range(len(test_images)):
        testset.append((test_images[i], test_labels[i]))
    return trainset, testset

def save_img(img: np.ndarray, label=None, path: str="img.png"):
    """
    @img: numpy.ndarray, shape=(3072,)
    """
    plt.figure()
    if img.shape != (3, 32, 32):
        img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    if label is not None:
        plt.title(label)
        
    plt.imsave(path, img)
    plt.close()

def get_backdoor_block(size: int = 4):
    block = np.zeros(shape=(3, size, size,), dtype=np.uint8)
    block[0, :, :] = 128
    block[1, :, :] = 128
    block[2, :, :] = 128
    return block

def add_backdoor_npimg(img: np.ndarray, x: int, y: int, backdoor: np.ndarray):
    img = img.reshape(3, 32, 32)
    x_len = backdoor.shape[1]
    y_len = backdoor.shape[2]
    img[:, x:x+x_len, y:y+y_len] = backdoor[:, x:x+x_len, y:y+y_len]
    return img.reshape(3072)

def add_backdoor_tsimg(img: torch.Tensor, x: int, y: int, backdoor: np.ndarray) -> torch.Tensor:
    # convert to np image to add backdoor
    img: np.ndarray =img.numpy()
    img *= 255 # scale to [0, 255]
    img = img.astype(np.uint8)
    # add backdoor
    x_len = backdoor.shape[1]
    y_len = backdoor.shape[2]
    img[:, x:x+x_len, y:y+y_len] = backdoor[:, x:x+x_len, y:y+y_len]
    # convert to torch tensor
    img = img.astype(np.float32)
    img /= 255 # scale to [0, 1]
    img = torch.from_numpy(img)

    return img

def load_cifar10(trainset, testset):
    pass

