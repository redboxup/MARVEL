import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import ResNet
from typing import cast, Callable, List, Optional, Tuple

from configurations import ExpConf
from losses.marvel import ProCoLoss

MODELS_REGISTRY = dict()


def register_model(class_):
    if 'Model' in class_.__name__:
        key_name = class_.__name__[:-5]
    else:
        raise NameError(f"{class_.__name__} should end with 'Model' keyword")
    MODELS_REGISTRY[key_name.lower()] = class_
    return class_


@register_model
class CrossEntropyModel(nn.Module):
    def __init__(self, configs: ExpConf, model_encoder: nn.Module):
        super().__init__()
        self.model_encoder = cast(ResNet, model_encoder)
        self.classifier = nn.Linear(
            in_features=self.model_encoder.num_features,
            out_features=configs.dataset.num_classes,
        )

    def forward_features(self, x: torch.Tensor):
        return self.model_encoder(x)

    def forward_classifier(self, x: torch.Tensor):
        return self.classifier(x)

    def forward(self, x: torch.Tensor):
        x = self.model_encoder(x)
        x = self.forward_classifier(x)
        return x

@register_model
class MarvelModel(nn.Module):
    def __init__(
            self,
            configs: ExpConf,
            model_encoder: nn.Module,
    ):
        super().__init__()
        num_experts = getattr(configs.trainer, 'num_experts', 3)
        self.backbone = cast(ResNet, model_encoder)

        self.proj_heads = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.backbone.num_features,
                    out_features=getattr(configs.trainer, 'feat_dim', 128),
                )
                for _ in range(num_experts)
            ]
        )

        self.proco_modules = nn.ModuleList(
            [
                ProCoLoss(
                    contrast_dim=getattr(configs.trainer, 'feat_dim', 128),
                    temperature=getattr(configs.trainer, 'temp', 1.0),
                    num_classes=configs.dataset.num_classes + 1,
                )
                for _ in range(num_experts)
            ]
        )
        self.fc = nn.Linear(
            in_features=self.backbone.num_features,
            out_features=2,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_proj_heads(self, feats: torch.Tensor) -> List[torch.Tensor]:
        return [proj(feats) for proj in self.proj_heads]

    def forward_proco(self,
                      x_list: List[torch.Tensor],
                      labels: Optional[torch.Tensor]
                    ) -> List[torch.Tensor]:
        if labels is None:
            return [proco(x) for x, proco in zip(x_list, self.proco_modules)]
        else:
            return [
                proco(x, labels)
                for x, proco in zip(x_list, self.proco_modules)
            ]

    def forward_fc(self, x: torch.Tensor):
        return self.fc(x)

    def forward(self,
                x: torch.Tensor,
                labels: Optional[torch.Tensor]
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.forward_features(x)
        x_list = self.forward_proj_heads(x)
        x_proco_list = self.forward_proco(x_list, labels)
        x_fc = self.forward_fc(x)

        return x_proco_list, x_fc


@register_model
class CEOutlierExposureModel(CrossEntropyModel):
    pass


if __name__ == '__main__':
    print(MODELS_REGISTRY)
