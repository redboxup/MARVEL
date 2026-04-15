import numpy as np
from typing import List, Union
from numpy.typing import NDArray
from torch.utils.data import Sampler
from collections import defaultdict


class BaseParetoSampler(Sampler):
    """
    BaseSampler to sample dataset indices bases on Pareto dist.
    """

    def __init__(self,
                 labels: Union[List[int], NDArray[np.int8]],
                 labels_to_include: List[int],
                 imbalnce_ratio: int,
                 random_seed: int,
                 ):
        self.labels = np.array(labels)
        self.labels_to_include = labels_to_include
        self.imbalance_ratio = imbalnce_ratio
        self.random_seed = random_seed
        self.indices = self._generate_indices()

    def _generate_indices(self) -> NDArray[np.int8]:
        raise RuntimeError(f'Implement this method in subclass')

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def generate_pareto_distribution(
    largest_class_size: int, num_classes: int, imbalance_ratio: int
):
    """
    Generate pareto distribution from dataset statistics

    Parameters:
        largest_class_size (int): Number of instances of the largest-class
        num_classes (int): Number of classes in the dataset
        imbalance_ratio (int) : imbalance ratio for the dataset
    """
    smallest_class_size = largest_class_size / imbalance_ratio

    # Calculate the ratio factor for the geometric series
    factor = (largest_class_size / smallest_class_size) ** (
        1 / (num_classes - 1)
    )

    class_instances = [
        round(largest_class_size / (factor**i)) for i in range(num_classes)
    ]

    class_instances[0] = largest_class_size

    return class_instances


class MulticlassParetoSampler(BaseParetoSampler):
    def _generate_indices(self):
        np.random.seed(self.random_seed)

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            if label in self.labels_to_include:
                label_to_indices[label].append(idx)

        class_sizes = {
            label: len(label_to_indices[label])
            for label in self.labels
        }

        largest_class_size = max(class_sizes.values())
        target_sizes = generate_pareto_distribution(
            largest_class_size=largest_class_size,
            num_classes=len(self.labels_to_include),
            imbalance_ratio=self.imbalance_ratio,
        )
        label_target_sizes = dict(zip(self.labels_to_include, target_sizes))

        selected_indices = []
        for label, target_size in label_target_sizes.items():
            indices = label_to_indices[label]
            num_to_select = min(target_size, len(indices))
            selected = np.random.choice(
                indices, size=num_to_select, replace=False
            )
            selected_indices.extend(selected)

        return np.array(selected_indices)


class MultilabelParetoSampler(BaseParetoSampler):
    def _generate_indices(self) -> NDArray[np.int8]:
        np.random.seed(self.random_seed)
        N = self.labels.shape[0]

        label_to_indices = defaultdict(list)
        for idx in range(N):
            for label in self.labels_to_include:
                if self.labels[idx][label] == 1:
                    label_to_indices[label].append(idx)

        class_sizes = {
            label: len(label_to_indices[label])
            for label in self.labels_to_include
        }
        largest_class_size = self.imbalance_ratio * min(class_sizes.values())

        target_sizes = generate_pareto_distribution(
            largest_class_size=largest_class_size,
            num_classes=len(self.labels_to_include),
            imbalance_ratio=self.imbalance_ratio,
        )
        label_target_sizes = dict(zip(self.labels_to_include, target_sizes))

        selected = set()
        for label, target_sizes in label_target_sizes.items():
            candidates = label_to_indices[label]
            num_to_select = min(len(candidates), target_sizes)
            sampled = np.random.choice(
                candidates, size=num_to_select, replace=False,
            )
            selected.update(sampled)

        return np.array(selected)
