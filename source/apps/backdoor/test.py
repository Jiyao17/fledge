

from img import *

data_path = "/home/tuo28237/projects/fledge/datasets/raw/cifar-10-batches-py"

trainset, testset = load_cifar10(data_path)

img_index = 1

show_img(trainset[img_index][0], trainset[img_index][1])

modified_img = add_red_block(trainset[img_index][0], 0, 0, 3)

show_img(modified_img, trainset[img_index][1])