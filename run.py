import os
import torch
import hydra
import dotenv
import pandas as pd
from pathlib import Path

from configurations import ExpConf
from trainers.base import BaseTrainer
from utils.logger import setup_logging
from utils.make import (
    make_transforms,
    make_datasets,
    make_dataloaders,
    make_model,
    make_optimizer,
    make_scheduler,
)


@hydra.main(config_name="expconf", version_base="1.3")
def main(configs: ExpConf):
    dotenv.load_dotenv(".env")
    configs.rootpath = Path(os.environ["ROOTPATH"])

    exp_id = "exp_00000000"
    experiment_dir = (
        configs.rootpath
        / "experiments_new"
        / configs.dataset.name
        / configs.trainer.name
        / exp_id
    )
    logs_dir = experiment_dir / "logs"
    results_dir = experiment_dir / "results"
    weights_dir = experiment_dir / "weights"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    configs.trainer.results_dir = results_dir
    configs.experiment_dir = experiment_dir

    logger = setup_logging(logs_dir, verbose=configs.verbose)
    logger.info("Logging initalised!")

    transforms = make_transforms(configs, logger)
    datasets = make_datasets(configs, logger)
    dataloaders = make_dataloaders(configs, datasets, logger)

    model = make_model(configs, logger)

    optimizer = make_optimizer(configs, model, logger)
    scheduler = make_scheduler(configs, optimizer, logger)

    trainer: BaseTrainer = BaseTrainer.registered_trainers[
        configs.trainer.name
    ](configs, model, dataloaders, transforms, optimizer, scheduler, logger)

    trainer.train()

    trainer.train_stage_two()

    try:
        state_dict = torch.load(
            weights_dir / f"epoch_{configs.trainer.epochs-1}.pth",
            weights_only=False,
        )
        logger.info("Loading weights corresponding to last epoch")
    except FileNotFoundError:
        state_dict = torch.load(
            weights_dir / "best_epoch.pth", weights_only=False
        )
        logger.info("Loading weights correpsonding to best validation metric")


    trainer.model.load_state_dict(state_dict["model"])

    results_dict = trainer.evaluate(split="test")
    msg = "Test Split: "
    msg += " ".join(
        [f"{key}: {value:.4f}" for key, value in results_dict.items()]
    )
    logger.info(msg)

    try:
        state_dict = torch.load(
            weights_dir / "best_epoch_stage2.pth", weights_only=False
        )
        trainer.model.load_state_dict(state_dict["model"])
    except FileNotFoundError:
        pass

    ood_dict = trainer.eval_ood()
    row_dict = results_dict | ood_dict

    try:
        results_df = pd.read_csv(results_dir / "all_results.csv")
        results_df = pd.concat(
            [results_df, pd.DataFrame([row_dict])],
            ignore_index=True,
        )
    except FileNotFoundError:
        results_df = pd.DataFrame([row_dict])

    results_df.to_csv(results_dir / "all_results.csv", index=False)
    print(results_df)
    trainer.create_tsne()
    print('tsne-created')

if __name__ == "__main__":
    main()
