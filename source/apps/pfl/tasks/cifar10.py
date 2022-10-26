
from torch.utils.data import Dataset
from ..dataset import DatasetPartitioner


class CIFAR10Partitioner(DatasetPartitioner):
    def __init__(self, dataset: Dataset, subset_num: int, data_num_range: tuple, alpha_range: tuple):
        super().__init__(dataset, subset_num, data_num_range, alpha_range)

    def get_label_types(self) -> 'list[int]':
        return list(range(10))

    def dataset_categorize(self) -> 'list[list[int]]':
        targets = self.dataset.targets
        categorized_indexes = [[] for i in range(10)]
        for i, target in enumerate(targets):
            categorized_indexes[target].append(i)
        return categorized_indexes

