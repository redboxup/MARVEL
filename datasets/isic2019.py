import torch
import pandas as pd
from PIL import Image
from typing import Any
from omegaconf.errors import ConfigAttributeError

from datasets.base import BaseDataset


class ISIC2019Dataset(BaseDataset):
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
            df = pd.read_csv(csv_path)[
                ["image"] + self.configs.dataset.labels_ood
            ]
            df = df[df[self.configs.dataset.labels_ood].sum(axis=1) >= 1]
        else:
            df = pd.read_csv(csv_path)[
                ["image"] + self.configs.dataset.labels
            ]

        if self.split == 'train':
            df = self._subsample_to_pareto_distribution(df)
        return df

    def __getitem__(self, index: int) -> Any:
        row = self.labels_df.iloc[index]
        image_path = self.images_dir / f"{row['image']}.png"
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
