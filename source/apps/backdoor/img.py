
import pickle


import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt


def read_cifar10(path: str):
    for i in range(1, 6):
        with open(path + f"/data_batch_{i}", "rb") as f:
            data = pickle.load(f, encoding="bytes")
            if i == 1:
                images = data[b"data"]
                labels = data[b"labels"]
            else:
                images = np.concatenate((images, data[b"data"]))
                labels = np.concatenate((labels, data[b"labels"]))

    with open(path + "/test_batch", "rb") as f:
        data = pickle.load(f, encoding="bytes")
        test_images = data[b"data"]
        test_labels = data[b"labels"]

    return images, labels, test_images, test_labels

def load_cifar10(path: str):
    """
    read cifar10 as trainset and testset
    dataset format: (img: numpy.ndarray, label)
    """
    images, labels, test_images, test_labels = read_cifar10(path)
    trainset = []
    testset = []
    for i in range(len(images)):
        trainset.append((images[i], labels[i]))
    for i in range(len(test_images)):
        testset.append((test_images[i], test_labels[i]))
    return trainset, testset

def show_img(img: np.ndarray, label: int=None):
    img = img.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.figure()
    plt.imshow(img)
    plt.savefig("test.png")
    # if label is not None:
    #     plt.title(label)
    # plt.show()
    # plt.close()

def add_red_block(img: np.ndarray, x: int, y: int, size: int):
    img = img.reshape(3, 32, 32)
    img[0, x:x+size, y:y+size] = 255
    img[1, x:x+size, y:y+size] = 0
    img[2, x:x+size, y:y+size] = 0
    return img.reshape(3072)

def add_backdoor(img: np.ndarray, x: int, y: int, backdoor: np.ndarray, size: int):

    img = img.reshape(3, 32, 32)
    backdoor = backdoor.reshape(3, 32, 32)
    img[:, x:x+size, y:y+size] = backdoor[:, x:x+size, y:y+size]
    return img.reshape(3072)
