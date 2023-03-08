import torch as th
import numpy as np
from sklearn.cluster import KMeans

import helpers
from lib import metrics
from models.base.base_model import BaseModel


@th.no_grad()
def spectral_clustering(aff, n_clusters, eps=1e-9):
    deg = aff.sum(dim=1)
    safe_deg = th.where(deg < eps, th.full_like(deg, eps), deg)
    deg_sqrt_inv = th.diag(1 / th.sqrt(safe_deg))
    lap_normed = th.eye(aff.size(0)).type_as(aff) - deg_sqrt_inv @ aff @ deg_sqrt_inv

    _, vec = th.linalg.eigh(lap_normed)
    vec = vec[:, :n_clusters]
    vec = helpers.npy(vec)
    pred = KMeans(n_clusters=n_clusters).fit_predict(vec)
    return pred


class BaseModelSpectral(BaseModel):
    @property
    def affinity(self):
        raise NotImplementedError()

    def dummy_output(self, n):
        return th.eye(self.cfg.n_clusters)[th.zeros(n).long()]

    def _val_test_step(self, batch, idx, prefix):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)

        # Only evaluate losses on full batches
        if labels.size(0) == self.cfg.batch_size:
            losses = self.get_loss()
            self._log_dict(losses, prefix=f"{prefix}_loss")

        return helpers.npy(labels)

    def _val_test_epoch_end(self, step_outputs, prefix):
        labels = np.concatenate(step_outputs, axis=0)
        aff = self.affinity.to(device="cpu")
        pred = spectral_clustering(aff, self.cfg.n_clusters)
        mtc = metrics.calc_metrics(labels=labels, pred=pred)
        self._log_dict(mtc, prefix=f"{prefix}_metrics")
