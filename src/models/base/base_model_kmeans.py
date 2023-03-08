import torch as th
import numpy as np
from sklearn.cluster import KMeans

import helpers
from lib import metrics
from models.base.base_model import BaseModel


class BaseModelKMeans(BaseModel):
    @property
    def eval_tensors(self):
        # Override to run k-means on different tensor.
        return th.stack(self.encoder_outputs, dim=1).detach()

    def dummy_output(self, n):
        return th.eye(self.cfg.n_clusters)[th.zeros(n).long()]

    def _val_test_step(self, batch, idx, prefix):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)

        # Only evaluate losses on full batches
        if labels.size(0) == self.cfg.batch_size:
            losses = self.get_loss()
            self._log_dict(losses, prefix=f"{prefix}_loss")

        return helpers.npy(labels), helpers.npy(self.eval_tensors)

    def _val_test_epoch_end(self, step_outputs, prefix):
        labels = np.concatenate([s[0] for s in step_outputs], axis=0)

        eval_tensors = np.concatenate([s[1] for s in step_outputs], axis=0)
        assert eval_tensors.ndim in {2, 3}
        if eval_tensors.ndim == 3:
            eval_tensors = np.concatenate([eval_tensors[:, v] for v in range(eval_tensors.shape[1])], axis=1)

        # FAISS-kmeans seems to be significantly worse than sklearn.
        # pred, *_ = helpers.faiss_kmeans(eval_tensors, self.cfg.n_clusters, n_iter=300, n_redo=10)
        pred = KMeans(n_clusters=self.cfg.n_clusters, n_init=10).fit_predict(eval_tensors)

        mtc = metrics.calc_metrics(labels=labels, pred=pred)
        self._log_dict(mtc, prefix=f"{prefix}_metrics")
