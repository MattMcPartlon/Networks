from typing import Optional, Dict

import torch
from torch import nn

get_eps = lambda x: torch.finfo(x.dtype).max  # noqa
get_max = lambda x: torch.finfo(x.dtype).eps  # noqa
MAX_FLOAT = 1e6
from einops import rearrange
from networks.common.helpers.torch_utils import ndim
import torch.nn.functional as F  # noqa
from torch import Tensor


def partition(lst, chunk):
    for i in range(0, len(lst), chunk):
        yield lst[i:i + chunk]


def get_scale(n: int) -> float:
    return 0.5 if n <= 15 else 1.24 * ((n - 15.0) ** (1. / 3.)) - 1.8


def compute_dists(coords: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.cdist(coords, coords + eps, p=2)


def add_batch(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 2:
        return x.unsqueeze(0)
    # TODO: batch size fixed at 1 currently
    assert len(x.shape) > 2 and x.shape[0] == 1
    return x


def get_loss_func(p: int, *args, **kwargs):
    if p == 1:
        return ClampedSmoothL1Loss(*args, **kwargs)
    elif p == 2:
        return ClampedMSELoss(*args, **kwargs)
    else:
        return lambda x, y: torch.mean(torch.pow(x - y, p) ** (1 / p))


def get_nbr_mask(n_nbrs: int, size: int):
    msk = torch.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if abs(i - j) <= n_nbrs and i != j:
                msk[i, j] = 1
    return msk == 1


def exclude_self(mask: torch.Tensor) -> torch.Tensor:
    n = mask.shape[0] if ndim(mask) == 2 else mask.shape[1]
    assert ndim(mask) < 4
    rng = torch.arange(n, device=mask.device)
    mask[..., rng, rng] = False
    return mask


def exclude_self_mask(size) -> torch.Tensor:
    return torch.eye(size) == 0


class ClampedSmoothL1Loss(torch.nn.Module):
    """Clamped smooth l1-loss"""

    def __init__(self, beta: float = 1, reduction: str = 'mean', min_clamp: float = 0, max_clamp: float = MAX_FLOAT):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.clamp = (min_clamp, max_clamp)

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        if self.beta < 1e-5:
            # avoid nan in gradients
            loss = torch.abs(pred - actual)
        else:
            n = torch.abs(pred - actual)
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        loss = loss.clamp(*self.clamp)
        if self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClampedMSELoss(torch.nn.Module):
    """Clamped MSE loss"""

    def __init__(
            self,
            beta=None,  # noqa
            reduction: str = 'mean',
            min_clamp: float = 0,
            max_clamp: float = 10,
            normalize=True):
        super().__init__()
        self.reduction = reduction
        self.clamp = (min_clamp, max_clamp)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        loss = (pred - actual) ** 2
        loss = torch.clamp(loss, self.clamp[0] ** 2, self.clamp[1] ** 2)
        # loss = loss / self.clamp[1] if self.normalize else loss
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -torch.sum(labels * nn.functional.log_softmax(logits, dim=-1), dim=-1)


def sigmoid_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_p = nn.functional.logsigmoid(logits)
    log_not_p = nn.functional.logsigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p


def get_centers(bin_edges: torch.Tensor):
    """Gets bin centers from the bin edges.

    Args:
      bin_edges: tensor of shape (num_bins + 1) defining bin edges

    Returns:
      bin_centers: [num_bins] the error bin centers.
    """
    centers = [(s + e) / 2 for s, e in zip(bin_edges[:-1], bin_edges[1:])]
    return torch.tensor(centers)


def compute_predicted_plddt(logits: torch.Tensor, bins: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes per-residue LDDT score in range [0,1].

    Args:
      logits: tensor of shape (b, n_res, n_bins)
      bins:

    Returns:
      plddt: tensor of shape (b, n_res,)
    """
    n_bins = logits.shape[-1]
    step = 1.0 / n_bins
    bins = bins if bins is not None else torch.arange(start=0.5 * step, end=1.0, step=step, device=logits.device)
    bins = rearrange(bins, "b -> () () b")
    return torch.sum(F.softmax(logits, dim=-1) * bins, dim=-1)


def compute_predicted_distance_error(
        logits: torch.Tensor,
        dist_bins: torch.Tensor) -> torch.Tensor:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: tensor of shape (n_res, n_res, d).
      dist_bins: tensor of shape (d)

    Returns:
      tensor of shape (n_res, n_res) containing expected
      (signed) distance error.

    """
    dist_probs = F.softmax(logits, dim=-1)
    return torch.sum(dist_probs * dist_bins.reshape(1, 1, dist_bins.shape[-1]), dim=-1)


def LDDT(predicted_coords, actual_coords, cutoff: float = 15., per_residue=True):
    # no batch dimension
    assert predicted_coords.shape[0] != 1
    assert ndim(predicted_coords) == ndim(actual_coords) == 2
    assert predicted_coords.shape == actual_coords.shape
    n = predicted_coords.shape[0]
    pred_dists = torch.cdist(predicted_coords, predicted_coords)
    actual_dists = torch.cdist(actual_coords, actual_coords)
    not_self = (1 - torch.eye(n, device=pred_dists.device)).bool()
    mask = torch.logical_and(pred_dists < cutoff, not_self).float()

    l1_dists = torch.abs(pred_dists - actual_dists).detach()

    scores = 0.25 * ((l1_dists < 0.5).float() +
                     (l1_dists < 1.0).float() +
                     (l1_dists < 2.0).float() +
                     (l1_dists < 4.0).float())

    dims = (0, 1) if not per_residue else (1,)
    eps = get_eps(l1_dists)
    scale = 1 / (eps + torch.sum(mask, dim=dims))
    scores = eps + torch.sum(scores * mask, dim=dims)

    return scale * scores


def uniform_layer(dim_in, dim_out, w_bounds, b_bounds):
    layer = nn.Linear(dim_in, dim_out)
    layer.weight.data.uniform_(*w_bounds)
    layer.bias.data.uniform_(*b_bounds)
    return layer


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)


def safe_detach_item(x) -> float:
    if torch.is_tensor(x):
        assert x.numel() == 1
        return x.detach().cpu().item()
    assert isinstance(x, int) or isinstance(x, float)
    return x


def to_info(baseline, actual, pred, loss_val: Optional[float] = None) -> Dict[str, float]:
    return dict(
        baseline=safe_detach_item(baseline),
        actual=safe_detach_item(actual),
        predicted=safe_detach_item(pred),
        loss_val=safe_detach_item(loss_val)
    )
