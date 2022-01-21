import torch
from torch import nn, Tensor
from networks.loss.utils import softmax_cross_entropy, partition
import torch.nn.functional as F  # noqa
from networks.common.utils import exists
from einops import rearrange
from typing import Dict, Optional


class PairDistLossNet(nn.Module):
    def __init__(self, dim_in, atom_tys, weight=1, step=0.25, d_min=2.5, d_max=20):
        super().__init__()
        assert len(atom_tys) % 2 == 0, f"must have even number of atom types, got: {atom_tys}"
        self.step, self.d_min, self.d_max = step, d_min, d_max
        self._bins = torch.arange(self.d_min, self.d_max + 2 * step, step=step)
        self.atom_ty_set = set(atom_tys)
        self.num_pair_preds = len(atom_tys) // 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, self.num_pair_preds * self._bins.numel()),
        )
        self.loss_fn = softmax_cross_entropy
        self.atom_tys = [(x, y) for x, y in partition(atom_tys, chunk=2)]
        self.weight = weight

    def bins(self, device) -> Tensor:
        # makes sure devices match
        if self._bins.device == device:
            return self._bins
        self._bins = self._bins.to(device)
        return self._bins

    def to_labels(self, dists: Tensor) -> Tensor:
        dists = torch.clamp(dists, self.d_min, self.d_max + self.step) - self.d_min
        labels = torch.round(dists / self.step).long()
        return F.one_hot(labels, num_classes=self._bins.numel())

    def forward(
            self,
            pair_state: Tensor,
            atom_coords: Dict[str, Tensor],
            atom_masks: Dict[str, Tensor],
            pair_mask: Optional[Tensor],
    ) -> Tensor:
        # TODO: atom_masks should be logical and of native and predicted masks?
        """
        :param pair_state:
        :param atom_coords: dict mapping from atom type to atom coordinates of shape (b,n,3)
        :param atom_masks:  dict mapping from atom type to atom coordinate mask of shape (b,n)
        :param pair_mask:
        :return:
        """
        a1_a2_dists, a1_a2_masks = [], []
        with torch.no_grad():
            for (a1, a2) in self.atom_tys:
                a1_coords, a2_coords = atom_coords[a1], atom_coords[a2]
                a1_a2_mask = torch.einsum("b i, b j -> b i j", atom_masks[a1], atom_masks[a2])
                a1_a2_dist = torch.cdist(a1_coords, a2_coords)
                a1_a2_mask = torch.logical_and(a1_a2_mask, pair_mask) if exists(pair_mask) else a1_a2_mask
                a1_a2_dists.append(a1_a2_dist)
                a1_a2_masks.append(a1_a2_mask)
            full_mask, full_dists = map(lambda x: torch.cat(x.unsqueeze(-1), dim=-1), (a1_a2_masks, a1_a2_dists))

        # get predictions
        labels = self.to_labels(full_dists)
        logits = rearrange(self.net(pair_state), "b n m (p d) -> b n m p d", p=len(self.atom_tys))
        return torch.mean(self.loss_fn(logits[full_mask], labels[full_mask]))
