import hydra
import dataclasses
from pathlib import Path
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from configs.datasets import (
    DatasetConf,
    ISICDatasetConf,
    OODDatasetConf,
    RFMiDDatasetConf,
    NCTCRCDatasetConf
)
from configs.trainers import (
    TrainConf,
    CrossEntropyConf,
    CEOutlierExposureConf,
    MarvelConf,
)
from configs.eval import EvalConf
from configs.models import ModelConf, ResNetConf
from configs.optimizer import OptimizerConf, AdamConf, SGDConf, WarmupConf
from configs.scheduler import (
    LRSchedulerConf,
    CosineSchedulerConf,
    NoSchedulerConf,
    LinearSchedulerConf,
)


@dataclass
class ExpConf:
    rootpath: Path = dataclasses.field(init=False)
    experiment_dir: Path = dataclasses.field(init=False)
    dataset: DatasetConf
    trainer: TrainConf
    model: ModelConf = dataclasses.field(default_factory=ResNetConf)
    optimizer: OptimizerConf = dataclasses.field(default_factory=AdamConf)
    scheduler: LRSchedulerConf = dataclasses.field(init=False)
    ood_dataset: OODDatasetConf = OODDatasetConf  # type: ignore
    warmup: WarmupConf = WarmupConf  # type: ignore
    eval: EvalConf = EvalConf  # type: ignore
    device: str = "cuda"
    verbose: bool = True


cs = ConfigStore.instance()
cs.store(name="expconf", node=ExpConf)
cs.store(name="isic2019", group="dataset", node=ISICDatasetConf)
cs.store(name="rfmid", group="dataset", node=RFMiDDatasetConf)
cs.store(name="nctcrc", group="dataset", node=NCTCRCDatasetConf)
cs.store(name="adam", group="optimizer", node=AdamConf)
cs.store(name="sgd", group="optimizer", node=SGDConf)
cs.store(name="cosine", group="scheduler", node=CosineSchedulerConf)
cs.store(name="linear", group="scheduler", node=LinearSchedulerConf)
cs.store(name="none", group="scheduler", node=NoSchedulerConf)
cs.store(name="crossentropy", group="trainer", node=CrossEntropyConf)
cs.store(name="ceoutlierexposure", group="trainer", node=CEOutlierExposureConf)
cs.store(name="resnet", group="model", node=ResNetConf)
cs.store(name="marvel", group="trainer", node=MarvelConf)


@hydra.main(config_name="expconf", version_base="1.3")
def main(config: ExpConf):
    print(config)


if __name__ == "__main__":
    main()
