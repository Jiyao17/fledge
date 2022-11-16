
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as tvtf

from img import *

data_path = "/home/tuo28237/projects/fledge/datasets/raw/"
# trainset, testset = read_cifar10(data_path)

img_index = 0
# raw_img, label = trainset[img_index]
# save_img(raw_img, label, "original.png")
backdoor = get_backdoor_block()
# modified_raw_img = add_backdoor_npimg(raw_img, 0, 0, backdoor)
# save_img(modified_raw_img, label, "modified.png")

dataset = CIFAR10(root=data_path, 
            transform=tvtf.ToTensor(), # scale to [0, 1]
            train=True,
            download=True)
img: torch.Tensor = dataset[img_index][0]
img: np.ndarray =img.numpy()
img *= 255 # scale to [0, 255]
img = img.astype(np.uint8)
save_img(img, None, "original_torch.png")
modified_img = add_backdoor_tsimg(img, 0, 0, backdoor)
save_img(modified_img, None, "modified_torch.png")

# backdoored_img = torch.from_numpy(modified_img) / 255

# diff = np.linalg.norm(img - raw_img.reshape(3, 32, 32))
# print(diff)
