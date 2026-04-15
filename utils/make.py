import timm
import torch
import torch.nn as nn
from typing import Union
from logging import Logger
import kornia.augmentation as K
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from kornia.augmentation import AugmentationSequential
from torch.utils.data import DataLoader, SubsetRandomSampler

from configurations import ExpConf
from datasets.base import BaseDataset
from trainers.models import MODELS_REGISTRY
from datasets.base import DATASETS_REGISTRY
from utils.reproducibility import deterministic_randperm


def make_model(configs: ExpConf, logger: Logger) -> nn.Module:
    """make function that will return the model required for the trainer

    Args:
        configs (ExpConf): Configuration file for the experiment

    Returns:
        model (nn.Module): Model for training the specific class
    """
    if configs.model.name == "resnet":
        model_encoder = timm.create_model(
            model_name=f"resnet{configs.model.variant}",
            pretrained=configs.trainer.pretrained,
            num_classes=0,
        )
        logger.info(f"Initialised a ResNet:{configs.model.variant} encoder")
    else:
        logger.error(f"[red] {configs.model.name} is not implemented[/red]")
        raise NotImplementedError(f"{configs.model.name} is not implemented")

    model_class = MODELS_REGISTRY[configs.trainer.name]
    model = model_class(configs, model_encoder)

    return model


def make_transforms(
    configs: ExpConf, logger: Logger
) -> dict[str, AugmentationSequential]:
    imsize = configs.trainer.img_size
    data_transforms = {}
    resize_and_normalize = AugmentationSequential(
        K.Resize(size=(imsize, imsize)),
        K.Normalize(mean=0.5, std=0.5),
    )
    if configs.trainer.transforms == "strong":
        data_transforms["train"] = AugmentationSequential(
            K.RandomResizedCrop(size=(imsize, imsize)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomGrayscale(p=0.2),
            K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            resize_and_normalize,
        )
        logger.info("Using Strong Augmentations for training")
    elif configs.trainer.transforms == "weak":
        data_transforms["train"] = AugmentationSequential(
            K.RandomResizedCrop(size=(imsize, imsize)),
            K.RandomHorizontalFlip(p=0.5),
            resize_and_normalize,
        )
        logger.info("Using Weak Augmentations for training")
    elif configs.trainer.transforms == "randaugment":
        data_transforms["train"] = AugmentationSequential(
            K.auto.RandAugment(n=2, m=9),  # type: ignore
            resize_and_normalize,
        )
        logger.info("Using Random Augmentations for training")

    for partition in ("val", "test"):
        data_transforms[partition] = resize_and_normalize

    return data_transforms


def make_datasets(configs: ExpConf, logger: Logger) -> dict[str, BaseDataset]:
    """creates ID and OOD datasets

    Args:
        configs (ExpConf): Configurations file for the experiment
        transforms (Dict[str, Compose]): Dictionary of transforms

    Returns:
        Dict[str, Dataset]: Dictionary of datasets
    """
    datasets = {}
    # Train, Val & Test split
    for split in ["train", "val", "test"]:
        dataset: BaseDataset = DATASETS_REGISTRY[configs.dataset.name](
            configs, split, transforms.ToTensor(),
        )
        datasets[split] = dataset
        logger.info(f"The {split}-distribution of labels is as follows")
        logger.info("\n%s", dataset.dataset_stats.to_string())

    # Novel Classes Dataset
    datasets["nov_path"] = DATASETS_REGISTRY[configs.dataset.name](
        configs, "nov_path", transforms.ToTensor(),
    )

    # Other OOD Classes
    for ood in ["nearood1", "nearood2", "nearood3", "farood", "aux"]:
        attr = (
            getattr(configs.ood_dataset, ood, None) or
            getattr(configs.dataset, ood)
        )
        dataset_cls = DATASETS_REGISTRY[attr]
        dataset = dataset_cls(configs, transforms.ToTensor())
        datasets[ood] = dataset
    return datasets


def make_dataloaders(
    configs: ExpConf, datasets: dict[str, BaseDataset], logger: Logger
) -> dict[str, DataLoader]:
    """Creates dataloaders for the datasets
    Args:
        configs (ExpConf): Configurations file for the experiment
        datasets (Dict[str, Dataset]): Dictionary of datasets

    Returns:
        Dict[str, DataLoader]: Dictionary of dataloaders
    """
    dataloaders = {}
    for split in datasets.keys():
        batch_size = (
            configs.trainer.batch_size if split == 'train'
            else configs.eval.batch_size
        )

        perm = deterministic_randperm(
            len(datasets["train"]),  # type: ignore
            seed=configs.trainer.seed,
        )
        sampler = (
            SubsetRandomSampler(perm.tolist())
            if split == 'aux'
            else None
        )
        num_workers = (
            configs.eval.num_workers if split == 'test' else 0
        )

        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )
    return dataloaders


def make_optimizer(
    configs: ExpConf, model: nn.Module, logger: Logger
) -> torch.optim.Optimizer:
    """make function to create an optimizer given the configurations

    Args:
        configs (ExpConf): Experiment configurations files
        model (BaseCLIP): model to be optimized over

    Returns:
        torch.optim.Optimizer: Optimizer object (e.g., Adam, sgd)
    """
    # get optimizer configurations
    optim_configs = {
        k: v
        for k, v in configs.optimizer.items()  # type: ignore
        if k != "name"
    }
    str_optim_configs = "_".join(
        f"{k}:{v}" for k, v in optim_configs.items()
    )

    # instantiate the optimizer
    if configs.optimizer.name == "adam":
        logger.info(f"Initialised ADAM optimizer with {str_optim_configs}")
        return torch.optim.Adam(
            model.parameters(), configs.trainer.learning_rate, **optim_configs
        )
    elif configs.optimizer.name == "sgd":
        logger.info(f"Initialised SGD optimizer with {str_optim_configs}")
        return torch.optim.SGD(
            model.parameters(), configs.trainer.learning_rate, **optim_configs
        )
    else:
        raise ValueError(f"Unsupported optimizer: {configs.optimizer.name}")


def make_scheduler(
    configs: ExpConf, optimizer: torch.optim.Optimizer, logger: Logger
) -> Union[_LRScheduler, lr_scheduler.SequentialLR]:
    """make function to create a learning rate scheduler

    Args:
        configs (ExpConf): Configurations file for experiment
        optimizer (torch.optim.Optimizer): Optimizer object

    Raises:
        ValueError: Only that scheduler mentioned is not supported

    Returns:
        _LRScheduler: learning rate scheduler for training
    """
    # get scheduler configurations
    sched_configs = {
        k: v
        for k, v in configs.scheduler.items()  # type: ignore
        if k != "name"
    }
    str_sched_configs = "_".join(
        [f"{k}:{v}" for k, v in sched_configs.items()]
    )

    # instantiate the scheduler
    if configs.scheduler.name == "linear":
        main_scheduler = lr_scheduler.LinearLR(optimizer, **sched_configs)
        logger.info(f"Initialised linear scheduler with {str_sched_configs}")
    elif configs.scheduler.name == "cosine":
        main_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, **sched_configs
        )
        logger.info(f"Initialised cosine scheduler with {str_sched_configs}")
    elif configs.scheduler.name == "none":
        main_scheduler = lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=1
        )
        logger.info(f"No learning rate scheduler being used")
    else:
        raise ValueError(f"Unsupported sceduler: {configs.scheduler.name}")

    if configs.warmup.use_warmup:
        # create warmup scheduler
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=configs.warmup.start_factor,
            end_factor=1.0,
            total_iters=configs.warmup.epochs,
        )
        # combine warmup_scheduler & main_scheduler
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[configs.warmup.epochs],
        )
        logger.info(f"Using warmup")
        return scheduler
    return main_scheduler  # type: ignore
