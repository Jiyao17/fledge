
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset

import numpy as np

def get_backdoor_block(size: int = 4, RGB: Sequence[int] = [128, 128, 128]):
    block = np.zeros(shape=(3, size, size,), dtype=np.uint8)
    for i in range(3):
        block[i, :, :] = RGB[i]
    return block

def add_backdoor_tsimg(img: torch.Tensor, x: int, y: int, backdoor: torch.Tensor):
    x_len = backdoor.shape[1]
    y_len = backdoor.shape[2]
    img[:, x:x+x_len, y:y+y_len] = backdoor[:, x:x+x_len, y:y+y_len]
    return img


def backdoor_dataset(dataset: Dataset,
        backdoor: torch.Tensor,
        backdoor_position: Sequence[int],
        backdoor_label: int,
        ):

    for i in range(len(dataset)):
        img, label = dataset[i]
        img = add_backdoor_tsimg(img, backdoor_position[0], backdoor_position[1], backdoor)
        label = backdoor_label
        dataset[i] = (img, label)
        
