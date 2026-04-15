import dataclasses
from dataclasses import dataclass

from .base import BaseConf


@dataclass
class ModelConf(BaseConf):
    name: str = dataclasses.field(init=False)
    variant: str = dataclasses.field(init=False)


@dataclass
class ResNetConf(ModelConf):
    name: str = "resnet"
    variant: str = "18"
