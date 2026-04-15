import dataclasses
from dataclasses import dataclass

from pathlib import Path


@dataclass
class TrainConf:
    name: str = dataclasses.field(init=False)
    img_size: int = 224
    batch_size: int = 256
    transforms: str = "weak"
    epochs: int = 50
    seed: int = 42
    learning_rate: float = 1e-4
    imbalance_ratio: int = 50

    save_interval: int = 25
    move_data_to_host: bool = True
    cache_dataset: bool = True
    results_dir: Path = dataclasses.field(init=False)

    pretrained: bool = True
    prec: str = "fp32"

    metric: str = dataclasses.field(init=False)


@dataclass
class CrossEntropyConf(TrainConf):
    name: str = "crossentropy"
    metric: str = "acc"


@dataclass
class CEOutlierExposureConf(TrainConf):
    name: str = "ceoutlierexposure"
    metric: str = "acc"
    oe: float = 1


@dataclass
class MarvelConf(TrainConf):
    name: str = "marvel"
    metric: str = "acc"
    marvel: float = 1.0
    crossentropy: float = 1.0