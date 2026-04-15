import torch
from typing import Dict
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder

from configurations import ExpConf
from datasets.base import DATASETS_REGISTRY


def DatasetFactory(name: str):
    class _Dataset(ImageFolder):
        def __init__(self, configs: ExpConf, transforms: Compose):
            _path = configs.rootpath / getattr(configs.ood_dataset, name)
            super().__init__(root=_path, transform=transforms)

        def __getitem__(  # type: ignore
            self, index: int
        ) -> Dict[str, torch.Tensor]:
            img, _ = super().__getitem__(index)
            return {"img": img}

    _Dataset.__name__ = f"{name.capitalize()}Dataset"
    DATASETS_REGISTRY[name.lower()] = _Dataset
    return _Dataset


# Using Factory create Datasets
ImageNet100 = DatasetFactory(name="imagenet100")  # aux dataset
MSCoCo5k = DatasetFactory(name="mscoco5k")
DFU = DatasetFactory(name="dfu")
PADUFES = DatasetFactory(name="padufes")
DeepDRiD = DatasetFactory(name="deepdrid")
OCT = DatasetFactory(name="oct")
CYTOLOGY = DatasetFactory(name='cytology')
MIHIC = DatasetFactory(name='mihic')
ISIC2019_CORR = DatasetFactory(name="isic2019_corr")
RFMID_CORR = DatasetFactory(name="rfmid_corr")
NCTCRC_CORR = DatasetFactory(name='nctcrc_corr')