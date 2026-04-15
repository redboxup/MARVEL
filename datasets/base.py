import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from omegaconf.errors import ConfigAttributeError

from configurations import ExpConf

DATASETS_REGISTRY: Dict[str, Any] = {}


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 configs: ExpConf,
                 split: str,
                 transforms: Compose,
                 ):
        super().__init__()
        self.configs = configs
        self.split = split
        self.transforms = transforms

        self.labels_df = self._get_labels()
        self.images_dir = self._get_images_dir()
        if split in ["train", "val", "test"]:
            self.dataset_stats = self._get_freq_distribution()

    def __len__(self):
        return len(self.labels_df)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement this method')

    @abstractmethod
    def _get_labels(self) -> pd.DataFrame:
        raise NotImplementedError('Subclass must implement this method')

    def _get_images_dir(self) -> Path:
        try:
            return (
                self.configs.rootpath
                / getattr(
                    self.configs.dataset, f"{self.split}_image_path"
                )
            )
        except ConfigAttributeError:
            return (
                self.configs.rootpath
                / getattr(
                    self.configs.dataset, "test_image_path"
                )
            )

    def _get_freq_distribution(self) -> pd.DataFrame:
        dataset_stats_df = (
            self.labels_df[self.configs.dataset.labels]
            .sum(axis=0)
            .astype(int)
            .reset_index()
            .rename(columns={"index": "class", 0: "count"})
            .sort_values(by="count", ascending=False)
        )
        dataset_stats_df["group"] = pd.cut(
            dataset_stats_df.index,
            bins=[
                0,
                self.configs.dataset.head,
                self.configs.dataset.mid,
                self.configs.dataset.tail,
            ],
            labels=["head", "mid", "tail"],
            include_lowest=True,
        )
        return dataset_stats_df

    def _generate_pareto_distribution(self,
                                      num_classes: int,
                                      imbalance_ratio: int,
                                      calibrating_label_size: int,
                                      is_calibrating_label_large: bool,
                                      ):
        if is_calibrating_label_large:
            largest_class_size = calibrating_label_size
            smallest_class_size = calibrating_label_size / imbalance_ratio
        else:
            smallest_class_size = calibrating_label_size
            largest_class_size = calibrating_label_size * imbalance_ratio

        # Calculate ratio factor for the geometric series
        factor = (largest_class_size / smallest_class_size) ** (
            1 / (num_classes - 1)
        )
        class_instances = [
            round(largest_class_size / (factor ** i))
            for i in range(num_classes)
        ]

        if is_calibrating_label_large:
            class_instances[0] = int(largest_class_size)
        else:
            class_instances[-1] = int(smallest_class_size)
        return class_instances

    def _subsample_to_pareto_distribution(
        self, labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        np.random.seed(self.configs.trainer.seed)

        current_label_distribution = (
            labels_df[self.configs.dataset.labels]
            .sum(axis=0)
            .sort_values(ascending=False)
            .astype(int)
            .to_dict()
        )

        if self.configs.dataset.calibrate_using_head:
            calibrating_label_size = max(current_label_distribution.values())
        else:
            calibrating_label_size = min(current_label_distribution.values())

        target_label_distribution = self._generate_pareto_distribution(
            len(self.configs.dataset.labels),
            self.configs.trainer.imbalance_ratio,
            calibrating_label_size,
            self.configs.dataset.calibrate_using_head,
        )
        selected_indices = set()  # only unique indices to be stored
        for label, label_size in zip(
            current_label_distribution.keys(), target_label_distribution
        ):
            candidate_indices = (
                labels_df[labels_df[label] == 1]
                .index
                .tolist()
            )
            subsample_size = min(len(candidate_indices), label_size)
            subsampled_indices = np.random.choice(
                candidate_indices, size=subsample_size, replace=False
            )
            selected_indices.update(subsampled_indices)

        return labels_df.iloc[list(selected_indices)]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__.endswith('Dataset'):
            name = cls.__name__.lower()[:-7]
        else:
            raise ValueError('Name of subclass should end with "Dataset"')
        DATASETS_REGISTRY[name] = cls
