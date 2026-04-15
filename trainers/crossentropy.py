import torch
import numpy as np
from sklearn import metrics
from pandas import DataFrame
from typing import Dict, Mapping, Any, Callable, Tuple

from trainers import detectors
from utils import scoring
from losses.outlierexposure import OELoss
from trainers.base import BaseTrainer
from trainers.detectors import DETECTOR_REGISTRY


class CrossEntropyTrainer(BaseTrainer):
    def get_losses(self) -> Mapping[str, torch.nn.Module]:
        return dict(ce=torch.nn.CrossEntropyLoss())

    def iter_dataloaders(self):
        return self.dataloaders["train"]

    def model_forward(self, batch_input):
        imgs = batch_input["img"].to(self.device)
        imgs = self.transforms["train"](imgs)  # kornia augmentations
        logits = self.model(imgs)
        return logits

    def compute_loss(self,
                     loss_name: str,
                     loss_fn: Callable,
                     batch_input: Any,
                     outputs: torch.Tensor,
                     ) -> torch.Tensor:
        labels = batch_input["label"].to(self.device)
        logits = outputs

        if loss_name == "ce":
            return loss_fn(logits, labels)
        else:
            raise ValueError(f"{loss_name} is not implemented")

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        self.model.eval()
        preds_list, labels_list = [], []

        if split == "val" and self.configs.trainer.move_data_to_host:
            iterable = self.val_batches
        else:
            iterable = self.dataloaders[split]

        with torch.inference_mode():
            for data in self.tqdm(iterable, desc=f"Evaluation on {split}"):
                imgs, labels = data["img"].to(self.device), data["label"]
                imgs = self.transforms["test"](imgs)
                logits: torch.Tensor = self.model.forward(imgs)
                preds = logits.argmax(dim=1, keepdim=True)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.numpy())

        preds_np, labels_np = np.array(preds_list), np.array(labels_list)
        cls_acc = (
            metrics.confusion_matrix(
                y_true=labels_np, y_pred=preds_np, normalize="true"
            )
            .diagonal()
            .tolist()
        )

        # Get head, mid & tail accuracy
        class_stats_df: DataFrame = (
            self.dataloaders["train"]
            .dataset
            .dataset_stats  # type: ignore
        )

        class_stats_df["accuracy"] = class_stats_df.index.map(
            lambda x: cls_acc[x]
        )

        grouped_class_accuracy = (
            class_stats_df.groupby("group", observed=False)["accuracy"]
            .mean()
            .to_dict()
        )

        if split == "test":
            self.logger.info(
                "Performance on Test Split\n%s",
                class_stats_df
                .sort_values(by='count', ascending=False)
                .to_string()
            )

        return dict(
            acc=metrics.accuracy_score(y_true=labels_np, y_pred=preds_np),
            f1=metrics.f1_score(
                y_true=labels_np, y_pred=preds_np, average="macro"
            ),
            bacc=metrics.balanced_accuracy_score(
                y_true=labels_np, y_pred=preds_np
            ),
            **grouped_class_accuracy,
        )

    def forward_feats_pass(
            self, name: str, desc: str
            ) -> Tuple[np.ndarray, np.ndarray]:
        feats_list, labels_list = [], []
        self.model.eval()
        with torch.inference_mode():
            for batch_input in self.tqdm(
                self.dataloaders[name],
                total=len(self.dataloaders[name]),
                desc=desc,
            ):
                images = self.transforms["test"](
                    batch_input["img"].to(self.device)
                )
                try:
                    labels = batch_input["labels"]
                except KeyError:
                    labels = torch.zeros(len(batch_input["img"]))

                feats = self.model.forward_features(images)  # type: ignore
                feats_list.extend(feats.cpu().numpy())
                labels_list.extend(labels)

        feats_np = np.array(feats_list)
        labels_np = np.array(labels_list)
        return feats_np, labels_np


    def eval_ood(self):
        self.model.eval()
        names_list = [
            "test", "nov_path", "nearood1", "nearood2", "nearood3", "farood",
        ]

        detector_name = getattr(self.configs, 'detector', 'msp')

        if detector_name in ['msp', 'mls', 'energy']:

            scoring_func = getattr(detectors, detector_name)
            dataset_logits = [
                self.forward_pass(name=name, desc=f"Forward pass on {name}")[0]
                for name in names_list
            ]
            ood_scores = [scoring_func(logits) for logits in dataset_logits]

        elif detector_name in ['mahalanobis', 'knn']:
            train_feats_labels = [
                self.forward_feats_pass(
                    'train', desc='Forward pass on train data'
                )
            ]
            detector = DETECTOR_REGISTRY[detector_name](
                train_feats_labels[0][0],
                train_feats_labels[0][1],
                self.configs.device,
            )
            dataset_feats_labels = [
                self.forward_feats_pass(
                    name, desc=f'Forward pass on {name}'
                )
                for name in names_list
            ]
            ood_scores = [
                detector.scoring_function(feats)
                for (feats, _) in dataset_feats_labels
            ]

        else:
            raise ValueError(f'{detector_name} is not supported')

        scoring_results = [
            scoring.get_measures(ood_scores[0], scores)
            for scores in ood_scores[1:]
        ]

        # Flatten evaluation results into a single dictionary,
        # prefixing each metric key
        # with the corresponding OOD dataset name eg:
        # {'farood_fpr95': 0.23, ...}
        return {
            f"{name}_{key}": value
            for name, result_dict in zip(names_list[1:], scoring_results)
            for key, value in result_dict.items()
        }

    def post_train_epoch(self):
        results_dict = self.evaluate(split="val")

        msg = "Val Split: "
        msg += " ".join(
            [
                f"{metric}: {results:.4f}:"
                for metric, results in results_dict.items()
            ]
        )
        self.logger.info(msg)

        if self.best_val_acc <= results_dict[self.configs.trainer.metric]:
            self.best_val_acc = results_dict[self.configs.trainer.metric]
            self.best_val_dict = results_dict
            results_dict["best_val_acc"] = self.best_val_acc

        return results_dict


class CEOutlierExposureTrainer(CrossEntropyTrainer):
    def get_losses(self) -> Mapping[str, torch.nn.Module]:
        return dict(
            ce=torch.nn.CrossEntropyLoss(),
            oe=OELoss(),
        )

    def iter_dataloaders(self):  # type: ignore
        yield from zip(self.dataloaders["train"], self.dataloaders["aux"])

    def model_forward(self, batch_input):
        imgs = self.transforms["train"](batch_input[0]["img"].to(self.device))
        auximgs = self.transforms["test"](
            batch_input[1]["img"].to(self.device)
        )
        logits = self.model(imgs)
        auxlogits = self.model(auximgs)
        return logits, auxlogits

    def compute_loss(self,
                     loss_name: str,
                     loss_fn: Callable,
                     batch_input: Any,
                     outputs
                     ) -> torch.Tensor:
        labels = batch_input[0]["label"].to(self.device)
        logits, aux_logits = outputs

        if loss_name == "ce":
            return loss_fn(logits, labels)

        elif loss_name == "oe":
            return loss_fn(aux_logits)

        else:
            raise ValueError(f"{loss_name} is not implemented")
