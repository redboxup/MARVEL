import dataclasses
from dataclasses import dataclass

from .base import BaseConf


@dataclass
class OptimizerConf(BaseConf):
    name: str = dataclasses.field(init=False)


@dataclass
class AdamConf(OptimizerConf):
    name: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class SGDConf(OptimizerConf):
    name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


@dataclass
class WarmupConf:
    use_warmup: bool = False
    epochs: int = 5
    start_factor: float = 1e-5
