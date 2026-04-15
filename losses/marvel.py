import torch
import torch.nn as nn
from scipy.special import ive
import torch.nn.functional as F
from typing import cast, List, Union


def miller_recurrence(nu, x):
    I_n = torch.ones(1, dtype=torch.float64).cuda()
    I_n1 = torch.zeros(1, dtype=torch.float64).cuda()

    Estimat_n = [nu, nu+1]
    scale0 = 0
    scale1 = 0
    scale = 0

    for i in range(2*nu, 0, -1):
        I_n_tem, I_n1_tem = 2*i/x*I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            if I_n.shape != I_n1.shape:
                I_n1 = I_n1.view(I_n.shape)
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (nu+1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == nu:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).cuda()
            I_n, I_n1 = 2*i/x*I_n + I_n1, I_n

        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == nu:
            Estimat_n[0] = I_n1
        elif i == (nu+1):
            Estimat_n[1] = I_n1

    ive0 = torch.special.i0e(x.cuda())

    Estimat_n[0] = torch.log(
        ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
    Estimat_n[1] = torch.log(
        ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

    return Estimat_n[0], Estimat_n[1]


class LogRatioC(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, k, p, logc):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        nu, nu1 = miller_recurrence((p/2 - 1).int(), k.double())
        # nu = log(ive(p/2-1, k)) = log(iv(p/2-1, k)) - k
        # nu1 = log(ive(p/2, k)) = log(iv(p/2, k)) - k

        tensor = nu + k - (p/2 - 1) * torch.log(k+1e-20) - logc
        # tensor = log(ive(p/2-1, k)) + k - (p/2 - 1) * log(k) - logc
        #        = log(iv(p/2-1, k)) - (p/2 - 1) * log(k) - logc
        #        = log(C(\tilde{\kappa})/C(\kappa))
        ctx.save_for_backward(torch.exp(nu1 - nu))

        # d/dk iv(p/2-1, k) = iv(p/2, k) + (p/2 - 1) / k * iv(p/2-1, k)
        #
        # d/dk tensor = (iv(p/2, k) + (p/2 - 1) / k * iv(p/2-1, k)) / iv(p/2-1, k) - (p/2 - 1) / k
        #             = iv(p/2, k) / iv(p/2-1, k)
        #             = exp(nu1 - nu)
        #

        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        grad = ctx.saved_tensors[0]
        # grad clip
        grad[grad > 1.0] = 1.0

        grad *= grad_output

        return grad, None, None


class MarvelLoss(nn.Module):
    def __init__(self, class_frequencies, tau_list=[0, 1, 2], eps=1e-9):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.register_buffer(
            'bsce_weight', torch.tensor(class_frequencies).float()
        )
        self.register_buffer('tau_list', torch.tensor(tau_list).float())
        self.num_experts = len(tau_list)
        self.eps = eps

    def get_default_bias(self, tau=1):
        prior = cast(torch.Tensor, self.bsce_weight)
        prior = prior / prior.sum()
        log_prior = torch.log(prior + self.eps)
        return tau * log_prior

    def forward(
            self,
            output_logits: List[torch.Tensor],
            targets: torch.Tensor,
        ):
        total_loss = 0
        expert_losses = {}
        tau_list = cast(list, self.tau_list)

        # output_logits expected to have shape
        # (num_experts, batch_size, num_classes)
        assert len(output_logits) == self.num_experts, (
            "Mismatch in number of experts"
        )
        assert output_logits[1].shape[0] == targets.shape[0], (
            "Batch size mismatch"
        )

        for idx in range(self.num_experts):
            bias = self.get_default_bias(
                tau_list[idx]
            ).to(output_logits[0].device)
            adjusted_logits = output_logits[idx] + bias
            loss = self.base_loss(adjusted_logits, targets)
            expert_losses[f'losses_e_{idx}'] = loss
            total_loss += loss

        total_loss = total_loss / self.num_experts

        return total_loss


class VMFPrototypeBank(nn.Module):
    """
    Keeps running estimates of class prototypes, concentration (kappa),
    and log normalization
    """
    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 device: Union[str, torch.device],
        ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.class_prototypes: torch.Tensor
        self.register_buffer(
            "class_prototypes", self._init_prototypes(device)
        )

        self.sample_counts: torch.Tensor
        self.register_buffer(
            "sample_counts", torch.zeros(num_classes, device=device)
        )

        self.kappa: torch.Tensor
        self.register_buffer(
            "kappa",
            torch.ones(num_classes, device=device) *
            feature_dim * 90.0 / 19.0
        )

        self.log_norm_const: torch.Tensor
        self.register_buffer(
            "log_norm_const", torch.zeros(num_classes, device=device)
        )
        self.update_normalizer()

    def _init_prototypes(
            self, device: Union[str, torch.device]
        ) -> torch.Tensor:
        return F.normalize(
            torch.randn(
                self.num_classes, self.feature_dim, device=device
            ),
            dim=1
        ) * 0.9

    def reset_statistics(self):
        """Reset statistics to their initial state"""
        device = self.class_prototypes.device

        self.class_prototypes.copy_(self._init_prototypes(device))
        self.sample_counts.zero_()
        self.kappa.fill_(self.feature_dim * 90.0 / 19.0)
        self.update_normalizer()

    @torch.no_grad()
    def update_prototypes(
            self, features: torch.Tensor, labels: torch.Tensor
        ) -> None:
        ncls, featdim = self.num_classes, self.feature_dim
        device = features.device
        features = F.normalize(features, dim=1)

        summed_features = torch.zeros(ncls, featdim, device=device)
        summed_features.index_add_(0, labels, features)

        counts_per_class = (
            torch.bincount(labels, minlength=ncls)
            .float()
            .to(device)
        )

        denom = counts_per_class.unsqueeze(1).clamp_min(1.0)

        batch_means = summed_features / denom  # shape: [num_cls, feat_dim]

        expanded_counts = counts_per_class.unsqueeze(1).expand(ncls, featdim)
        sample_counts = self.sample_counts.unsqueeze(1).expand(ncls, featdim)

        weight = expanded_counts / (expanded_counts + sample_counts)
        weight = torch.nan_to_num(weight, nan=0.0)

        prototypes = (
            self.class_prototypes * (1 - weight) + batch_means * weight
        )
        self.class_prototypes.copy_(prototypes)

        self.sample_counts.add_(counts_per_class)

    @torch.no_grad()
    def update_kappa(self) -> None:
        R = torch.linalg.norm(self.class_prototypes, dim=1)
        # TODO: try different approximation for kappa
        kappa_est = self.feature_dim * R / (1 - R**2)
        self.kappa.copy_(torch.clamp(kappa_est, min=0.0, max=1e5))
        self.update_normalizer()

    @torch.no_grad()
    def update_normalizer(self):
        nu = self.feature_dim / 2 - 1
        device = self.kappa.device
        kappa = self.kappa.cpu().numpy()

        # Exponentially scaled modified Bessel function of the first kind
        scaled_bessel = torch.from_numpy(ive(nu, kappa)).to(device=device)

        # log(Iν(κ)) = log(ive(nu, κ)) + κ
        log_bessel = torch.log(scaled_bessel + 1e-300) + self.kappa

        # logC_d(κ) = log(I_{d/2-1}(κ)) - (d/2 - 1) * log(κ)
        log_c = log_bessel - nu * torch.log(self.kappa + 1e-300)

        self.log_norm_const.copy_(log_c)


class ProCoLoss(nn.Module):
    def __init__(self, contrast_dim, temperature=1.0, num_classes=1000):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = contrast_dim
        self.estimator_old = VMFPrototypeBank(
            self.feature_num, num_classes, device='cuda',
        )
        self.estimator = VMFPrototypeBank(
            self.feature_num, num_classes, device='cuda'
        )

    def forward(self, features, labels=None):
        batch_size = features.size(0)
        N = batch_size

        if labels is not None:
            total_features = features.clone()
            total_labels = labels.clone()

            self.estimator_old.update_prototypes(
                total_features.detach(), total_labels
            )
            self.estimator.update_prototypes(
                total_features.detach(), total_labels
            )
            self.estimator_old.update_kappa()

        Ave = self.estimator_old.class_prototypes.detach()
        Ave_norm = F.normalize(Ave, dim=1)

        logc = self.estimator_old.log_norm_const.detach()

        kappa = self.estimator_old.kappa.detach()

        tem = kappa.reshape(-1, 1) * Ave_norm

        tem = (
            tem.unsqueeze(0) + features[:N].unsqueeze(1) / self.temperature
        )

        kappa_new = torch.linalg.norm(tem, dim=2)

        contrast_logits = LogRatioC.apply(
            kappa_new, torch.tensor(self.estimator.feature_dim), logc
        )

        return contrast_logits
