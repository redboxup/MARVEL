import dataclasses
from dataclasses import dataclass

from .base import BaseConf


@dataclass
class LRSchedulerConf(BaseConf):
    name: str = dataclasses.field(init=False)


@dataclass
class CosineSchedulerConf(LRSchedulerConf):
    name: str = "cosine"
    T_max: int = 50  # Maximum number of iterations
    eta_min: float = 1e-5  # Minimum learning rate
    last_epoch: int = -1  # Last epoch for restart


@dataclass
class LinearSchedulerConf(LRSchedulerConf):
    name: str = "linear"
    total_iters: int = 100  # Number of iterations for decay
    start_factor: float = 1.0  # Starting factor of LR
    end_factor: float = 0.0  # Ending factor of LR


@dataclass
class NoSchedulerConf(LRSchedulerConf):
    name: str = "none"
