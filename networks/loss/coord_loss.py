import torch
import torch.nn as nn
from einops import rearrange, repeat  # noqa
from networks.common.utils import calc_tm_torch, exists
from networks.loss.utils import get_loss_func
from torch import Tensor

BETA = 0.25
MAX_DIST_CLAMP, MIN_DIST_CLAMP = 10, 0
MAX_COORD_CLAMP, MIN_COORD_CLAMP = 10, 0

flatten_coords = lambda crds: rearrange(crds, "b n a c-> b (n a) c")
ALIGN = True

get_eps = lambda x: torch.finfo(x.dtype).eps  # noqa
zero_loss = lambda device: torch.zeros(1, device=device, requires_grad=True)
DEFAULT_SEPS = [1, 5, 13]
to_res_rel_coords = lambda x, y: rearrange(x, "b n a c -> b n a () c") - rearrange(y, "b n a c -> b n () a c")
BB_LDDT_THRESHOLDS, SC_LDDT_THRESHOLDS = [0.5, 1, 2, 4], [0.25, 0.5, 1, 2]
to_rel_dev_coords = lambda x: rearrange(x, "b n a c -> b (n a) () c") - rearrange(x, "b n a c -> b () (n a) c")


def per_res_coord_mean(x: Tensor, mask: Tensor, exclude_missing=True) -> Tensor:
    """

    :param x: shape (b,n,a,3)
    :param mask: shape (b,n,a)
    :param exclude_missing:
    :return:
    """
    if mask is None:
        return torch.mean(x)
    atoms_per_res = mask.sum(dim=-1)
    exclude_mask = atoms_per_res > 0 if exclude_missing else None
    x = x.masked_fill(mask.unsqueeze(-1), value=0.)
    per_res_mean = x.sum(dim=(-1, -2)) / torch.clamp_min(atoms_per_res, 1.)
    scale = torch.sum(exclude_mask, dim=-1, keepdim=True) if exists(exclude_mask) else None
    return torch.sum(per_res_mean / scale) if exists(scale) else torch.mean(per_res_mean)


class TMLoss(nn.Module):
    def __init__(self, weight=1, eps=1e-8):
        super(TMLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, predicted_coords: torch.Tensor, actual_coords: torch.Tensor) -> torch.Tensor:  # noqa
        dev = torch.norm((predicted_coords - actual_coords) + self.eps, dim=-1)
        assert dev.shape[0] == actual_coords.shape[0]
        return -calc_tm_torch(dev) * self.weight


class CoordDeviationLoss(nn.Module):
    """
    Computes the loss between two sets of coordinates by minimizing
    l_p distance, for p = 1, 2, ...
    """

    def __init__(self,
                 weight: float = 1,
                 p: int = 1,
                 ):
        super(CoordDeviationLoss, self).__init__()
        self.loss_fn = get_loss_func(p, beta=BETA, min_clamp=MIN_COORD_CLAMP, max_clamp=MAX_COORD_CLAMP,
                                     reduction="none")
        self.weight = weight

    def forward(self, predicted: Tensor, actual: Tensor, mask: Tensor) -> Tensor:
        return per_res_coord_mean(self.loss_fn(predicted, actual), mask=mask)


class SCDihedralLoss(nn.Module):
    """
    Loss on predicted sidechain dihedral angles
    """

    def __init__(self, *args, **kwargs):  # noqa
        super(SCDihedralLoss, self).__init__()

    def forward(self, output: Output, baseline=False) -> Tensor:  # noqa
        native_chis, pred_chis, _, mask = output.chi_angles_n_mask
        if not baseline:
            return torch.mean((1 - torch.cos(pred_chis[mask] - native_chis.detach()[mask])))
        return torch.mean((1 - torch.cos(-native_chis.detach()[mask])))


class CoordLossTys:
    DIST_L2 = 'dist_l2'
    DIST_L1 = 'dist_l1'
    DIST_INV = 'inverse_dist'
    COORD_L2 = 'coord_l2'
    COORD_TM = 'coord_tm'
    COORD_L1 = 'coord_l1'
    L1_REL_DEV = 'l1_rel_dev'
    L2_REL_DEV = 'l2_rel_dev'
    L1_REL_DEV_PROX = "l1_rel_dev_prox"
    L2_REL_DEV_PROX = "l2_rel_dev_prox"
    LDDT_PROX = "lddt_prox"
    CONSECUTIVE_DIST = 'consecutive_dist'
    CONSECUTIVE_ANGLE = 'consecutive_angle'
    PER_RES_REL_DEV = "per_res_rel_dev"
    SC_DIHEDRAL = "sc_dihedral"
    VDW = "vdw_rep"

    def all_tys(self):  # noqa
        return {
            self.DIST_L1,
            self.DIST_L2,
            self.DIST_INV,
            self.COORD_L1,
            self.COORD_TM,
            self.COORD_L2,
            self.CONSECUTIVE_ANGLE,
            self.CONSECUTIVE_DIST,
            self.L1_REL_DEV,
            self.L2_REL_DEV,
            self.LDDT_PROX,
            self.L1_REL_DEV_PROX,
            self.L2_REL_DEV_PROX,
            self.PER_RES_REL_DEV,
            self.SC_DIHEDRAL,
            self.VDW,
        }


COORD_LOSS_TY_DESCS = [
    (CoordLossTys.DIST_L2, "invariant distance loss - l2 norm on inter atom distances"),
    (CoordLossTys.DIST_L1, "invariant distance loss - l1 norm on inter atom distances"),
    (CoordLossTys.DIST_INV, "invariant distance loss - 1/s*l1 on inter atom distances"),
    (CoordLossTys.COORD_L2, "RMSD between model and native (alignment = RMSD)"),
    (CoordLossTys.COORD_TM, "TM loss between model and native (alignment = TM-Align)"),
    (CoordLossTys.COORD_L1, "L1 loss between model and native (alignment = TM-Align)"),
    (CoordLossTys.L2_REL_DEV, "Per-residue-pair loss applied to relative positions"
                              " between coordinates (similar to AF2 pairwise"
                              " orientation loss)"),
]


def get_coord_loss_func(ty, **kwargs):
    if ty == CoordLossTys.COORD_L2:
        return CoordDeviationLoss(p=2, **kwargs)

    if ty == CoordLossTys.COORD_L1:
        return CoordDeviationLoss(p=1, **kwargs)

    if ty == CoordLossTys.SC_DIHEDRAL:
        return SCDihedralLoss(**kwargs)

    return None
