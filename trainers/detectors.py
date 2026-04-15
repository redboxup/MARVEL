import torch
import faiss
import numpy as np
from scipy import special
from typing import Union, List
from sklearn.covariance import EmpiricalCovariance

def msp(logits: np.ndarray) -> np.ndarray:
    return np.max(special.softmax(logits, axis=1), axis=1)


def mls(logits: np.ndarray) -> np.ndarray:
    return np.max(logits, axis=1)


def energy(logits: np.ndarray) -> np.ndarray:
    return special.logsumexp(logits, axis=1)  # type: ignore

class MahalanobisDistance:
    def __init__(self,
                 id_feats: np.ndarray,
                 id_labels: np.ndarray,
                 device: Union[str, torch.device]
                ):
        self.device = device
        self.id_feats = id_feats
        self.id_labels = id_labels
        self._setup()

    def _setup(self):
        class_means, centered_feats = [], []
        for label in np.unique(self.id_labels):
            feats_c = self.id_feats[self.id_labels == label]
            mean_c = feats_c.mean(axis=0)
            class_means.append(mean_c)
            centered_feats.append(feats_c - mean_c)

        centered_feats = np.vstack(centered_feats)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(centered_feats)

        # Convert to torch
        self.mean_t = torch.from_numpy(np.stack(class_means)).to(self.device)
        self.prec_t = torch.from_numpy(ec.precision_).to(self.device)

    def scoring_function(self, feats: torch.Tensor) -> np.ndarray:
        feats = torch.from_numpy(feats).to(self.device)
        # diff: torch.Tensor(n_samples, n_classes, d)
        diff = feats.unsqueeze(1) - self.mean_t.unsqueeze(0)
        m_dist = torch.einsum('ncd,dd->ncd', diff, self.prec_t) * diff
        # dist: torch.Tensor(n_samples, n_classes)
        dist = m_dist.sum(dim=-1)
        min_dist, _ = dist.min(dim=-1)
        return (-min_dist).cpu().numpy()


class KNNDistance:
    def __init__(self,
                 id_feats: np.ndarray,
                 id_labels: np.ndarray,
                 device: Union[str, torch.device],
                ):
        self.device = device
        self.id_feats = id_feats
        self.id_labels = id_labels
        self._setup()

    def _setup(self):
        self.index = faiss.IndexFlatL2(self.id_feats.shape[1])
        self.index.add(
            np.ascontiguousarray(self.id_feats, dtype=np.float32)
            )  # type: ignore

    def scoring_function(
            self, feats: np.ndarray, K: int = 500
        ) -> np.ndarray:
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        D, _ = self.index.search(feats, int(K))  # type: ignore
        return D[:, -1]

DETECTOR_REGISTRY = {
    'mahalanobis': MahalanobisDistance,
    'knn': KNNDistance,
}