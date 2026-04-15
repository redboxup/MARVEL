import torch
import pandas as pd
from PIL import Image
from typing import Any
from omegaconf.errors import ConfigAttributeError

from datasets.base import BaseDataset


class RFMiDDataset(BaseDataset):
    def _get_labels(self) -> pd.DataFrame:
        try:
            csv_path = (
                self.configs.rootpath
                / getattr(self.configs.dataset, f"{self.split}_csv_path")
            )
        except ConfigAttributeError:
            csv_path = (
                self.configs.rootpath
                / getattr(self.configs.dataset, "test_csv_path")
            )

        if self.split == 'nov_path':
            labels_df = pd.read_csv(csv_path)[
                ["ID"] + self.configs.dataset.labels_ood
            ]
            labels_df = labels_df[
                labels_df[self.configs.dataset.labels_ood].sum(axis=1) >= 1
            ]
        else:
            labels_df = pd.read_csv(csv_path)[
                ["ID"] + self.configs.dataset.labels
            ]
        if self.split == 'train':
            labels_df = self._remove_multilabel(labels_df)
            labels_df = self._subsample_to_pareto_distribution(labels_df)

        elif self.split in ['val', 'test']:
            labels_df = self._remove_multilabel(labels_df)

        return labels_df

    def __getitem__(self, index: int) -> Any:
        row = self.labels_df.iloc[index]
        image_path = self.images_dir / f"{row['ID']}.png"
        img = Image.open(image_path)
        img = self.transforms(img)

        if self.split == "nov_path":
            class_label = (
                row[self.configs.dataset.labels_ood]
                .to_numpy()
                .argmax()
            )

        else:
            class_label = (
                row[self.configs.dataset.labels]
                .to_numpy()
                .argmax()
            )

        return {"img": img, "label": torch.tensor(class_label)}

    def _remove_multilabel(self, df: pd.DataFrame):
        label_sum = df[self.configs.dataset.labels].sum(axis=1)
        single_label_df = df[label_sum == 1].reset_index(drop=True)
        return single_label_df