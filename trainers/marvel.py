import torch
import numpy as np
from sklearn import metrics
from pandas import DataFrame
from scipy.special import softmax
from torch.nn.modules import Module
from typing import (
    Any, Mapping, Callable, Dict, List, Optional, Union, TypeVar, cast
)

from utils import scoring
from trainers import detectors
from datasets.base import BaseDataset
from trainers.base import BaseTrainer
from losses.marvel import MarvelLoss
from trainers.models import MarvelModel

T = TypeVar('T', torch.Tensor, np.ndarray)

class MarvelTrainer(BaseTrainer):
    def get_losses(self) -> Mapping[str, Module]:
        base_dataset = cast(BaseDataset, self.dataloaders["train"].dataset)
        class_frequencies = base_dataset.dataset_stats["count"].to_list()
        class_frequencies.append(sum(class_frequencies))
        self.class_frequencies = class_frequencies
        self.prior = (
            torch.tensor(class_frequencies[:-1]) / sum(class_frequencies[:-1])
        )

        return dict(
            marvel = MarvelLoss(
                class_frequencies=class_frequencies,
                tau_list=getattr(self.configs, 'tau_list', [0, 1, 2]),
            ),
            crossentropy = torch.nn.CrossEntropyLoss(),
        )

    # def calibrate_logits(
    #         self, logits: T
    #     ) -> T:
    #     if isinstance(logits, np.ndarray):
    #         prior = np.concatenate((self.prior, np.array([1.0])))
    #         return logits - np.log(prior + 1e-9)  # type: ignore[return-value]

    #     elif isinstance(logits, torch.Tensor):
    #         prior = torch.concat((self.prior, torch.tensor([1.0]))).cpu()
    #         return logits.detach().cpu() - torch.log(prior + 1e-9)  # type: ignore[return-value]

    #     else:
    #         raise ValueError("Unsupported type for prior")


    def iter_dataloaders(self):
        yield from zip(self.dataloaders["train"], self.dataloaders["aux"])

    def _generate_randperm(self,
                           length: int,
                           seed: Optional[int] = None
                            ) -> torch.Tensor:
        g = torch.Generator()
        if seed:
            g.manual_seed(seed)
        return torch.randperm(n=length, generator=g)

    def model_forward(self, batch_input):
        id_imgs = self.transforms["train"](
            batch_input[0]["img"].to(self.device)
        )
        cls_labels = batch_input[0]["label"].to(self.device)

        aux_imgs = self.transforms["train"](
            batch_input[1]["img"].to(self.device)
        )
        aux_labels = (
            torch.ones(cls_labels.size(), dtype=torch.long) *
            self.configs.dataset.num_classes
        ).to(device=self.device)

        perm = self._generate_randperm(length=cls_labels.size()[0] * 2)

        all_imgs = torch.cat([id_imgs, aux_imgs], dim=0)[perm]
        all_labels = torch.cat([cls_labels, aux_labels], dim=0)[perm]
        id_ood_labels = torch.cat([
            torch.zeros(cls_labels.size(), dtype=torch.long),
            torch.ones(aux_labels.size(), dtype=torch.long),
        ], dim=0).to(device=self.device)[perm]

        self.model = cast(MarvelModel, self.model)
        feats = self.model.forward_features(all_imgs)
        projs_list = self.model.forward_proj_heads(feats)

        all_proco_logits_list = self.model.forward_proco(
            projs_list, all_labels
        )
        all_fc = self.model.forward_fc(feats)

        return all_proco_logits_list, all_fc, all_labels, id_ood_labels

    def compute_loss(self,
                     loss_name: str,
                     loss_fn: Callable,
                     batch_input: Any,
                     outputs) -> torch.Tensor:
        bsz = batch_input[0]["img"].size()[0]
        all_proco_logits_list, all_fc, all_labels, id_ood_labels = outputs

        if loss_name == "marvel":
            loss = loss_fn(all_proco_logits_list, all_labels)
            return loss

        elif loss_name == "crossentropy":
            loss = loss_fn(all_fc, id_ood_labels)
            return loss

        else:
            raise ValueError(f"{loss_name} is not supported")

    def get_default_bias(self, tau: int=1) -> torch.Tensor:
        prior = torch.tensor(self.class_frequencies).to(self.device)
        prior = prior / prior.sum()
        log_prior = torch.log(prior + 1e-9)
        return tau * log_prior

    def get_bias_list(
            self, tau_list: List[int] = [0, 1, 2]
        ) -> List[torch.Tensor]:
        return [self.get_default_bias(tau) for tau in tau_list]

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        self.model.eval()
        preds_list, labels_list = [], []

        if split == "val" and self.configs.trainer.move_data_to_host:
            iterable = self.val_batches
        else:
            iterable = self.dataloaders[split]

        # get biases to adjust logits
        bias_list = self.get_bias_list(tau_list = [0, 1, 2])

        with torch.inference_mode():
            for data in self.tqdm(iterable, desc=f"Evaluation on {split}"):
                imgs, labels = data["img"].to(self.device), data["label"]
                imgs = self.transforms["test"](imgs)
                logits_list: List[torch.Tensor] = self.model(imgs, None)[0]
                adjusted_logits_list = [
                    logits # + bias
                    # self.calibrate_logits(logits)
                    for logits, bias in zip(logits_list, bias_list)
                ]
                probs_list = [
                    torch.softmax(logits, dim=1)
                    # logits
                    for logits in adjusted_logits_list
                ]
                probs = torch.stack(probs_list, dim=0).mean(dim=0)

                preds = probs[:, :-1].argmax(dim=1, keepdim=True)
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

    def ood_logit_aggregation(
            self, name: str, desc: str,
    ):
        num_experts = getattr(self.configs.trainer, 'num_experts', 3)
        expert_logits_list: List[List[torch.Tensor]] = [
            [] for _ in range(num_experts)
        ]
        ood_logits_list, labels_list = [], []

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
                    labels = batch_input["label"]
                except KeyError:
                    labels = torch.zeros(len(batch_input["img"]))

                expert_logits, ood_logits = self.model(images, None)

                for i in range(num_experts):
                    expert_logits_list[i].extend(
                        expert_logits[i].cpu().numpy()
                        # self.calibrate_logits(expert_logits[i].cpu().numpy())
                    )
                ood_logits_list.extend(ood_logits.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        ood_logits_np = np.array(ood_logits_list)
        expert_logits_np = [
            np.array(expert_logits) for expert_logits in expert_logits_list
        ]
        # labels_np = np.array(labels_list)

        # logit aggregation strategy
        # confidence scores
        expert_probs_np = [softmax(logits) for logits in expert_logits_np]
        probs_np = np.stack(expert_probs_np, axis=0).mean(axis=0)
        ood_logits_np = softmax(ood_logits_np, axis=1)
        ood_ = np.stack([probs_np[:, -1], ood_logits_np[:, -1]], axis=0).mean(axis=0)

        return - ood_

    def forward_feats(self, name: str, desc: str):
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

                self.model = cast(MarvelModel, self.model)
                feats = self.model.forward_features(images)
                feats_list.extend(feats.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        feats_np = np.array(feats_list)
        labels_np = np.array(labels_list)
        return feats_np, labels_np


    def eval_ood(self):
        self.model.eval()
        names_list = [
            "test", "nov_path", "nearood1", "nearood2", "nearood3", "farood",
        ]

        confidence_ood_scores = [
            self.ood_logit_aggregation(name, desc=f'Forward pass on {name}')
            for name in names_list
        ]

        # id_feats, id_labels = self.forward_feats(
        #     name='train', desc='Forward feats on train'
        # )

        # feats_list = [
        #     self.forward_feats(name, desc=f"Forward feats on {name}")[0]
        #     for name in names_list
        # ]

        # knn_detector = detectors.KNNDistance(
        #     id_feats = id_feats, id_labels = id_labels, device=self.device,
        # )

        # distance_ood_scores = [
        #     knn_detector.scoring_function(feats)
        #     for feats in feats_list
        # ]

        # ood_scores = [
        #     distance_scores
        #     for confidence_scores, distance_scores in
        #     zip(confidence_ood_scores, distance_ood_scores)
        # ]
        ood_scores = [
            confidence_scores
            for confidence_scores in zip(confidence_ood_scores)
        ]

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
        # test_results_dict = self.evaluate(split='test')

        msg = "Val Split: "
        msg += " ".join(
            [
                f"{metric}: {results:.4f}:"
                for metric, results in results_dict.items()
            ]
        )

        self.logger.info(msg)

        # msg = "Test Split: "
        # msg += " ".join(
        #     [
        #         f"{metric}: {results:.4f}:"
        #         for metric, results in test_results_dict.items()
        #     ]
        # )

        # self.logger.info(msg)

        if self.best_val_acc <= results_dict[self.configs.trainer.metric]:
            self.best_val_acc = results_dict[self.configs.trainer.metric]
            self.best_val_dict = results_dict
            results_dict["best_val_acc"] = self.best_val_acc

        return results_dict
