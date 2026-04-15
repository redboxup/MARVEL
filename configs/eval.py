from dataclasses import dataclass

from .base import BaseConf


@dataclass
class EvalConf(BaseConf):
    batch_size: int = 256
    topk: int = 5
    num_workers: int = 8
