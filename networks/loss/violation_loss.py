from einops import repeat, rearrange  # noqa
from common.sidechain_constants import ALL_ATOMS, ALL_ATOM_POSNS
from networks.common.constants import MAX_SEQ_LEN
import torch
import torch.nn as nn
from data.data_types import Output

vdw_radius = dict(C=1.7, O=1.52, N=1.55, S=1.8)
atom_radii = {}
for atom in ALL_ATOMS:
    for ty in vdw_radius:
        if ty in atom:
            atom_radii[atom] = vdw_radius[ty]

to_rel_coord = lambda x: rearrange(x, "n c -> n () c") - rearrange(x, "n c-> () n c")
outer_sum = lambda x: rearrange(x, "i -> i ()") + rearrange(x, "i -> () i")


def _init_table():
    vdw_radii = torch.zeros(len(ALL_ATOMS))
    for ai in ALL_ATOMS:
        i = ALL_ATOM_POSNS[ai]
        vdw_radii[i] = atom_radii[ai]
    return repeat(vdw_radii, "i-> n i", n=MAX_SEQ_LEN)


class VDWRepulsiveLoss(nn.Module):

    def __init__(self, **kwargs): # noqa
        super(VDWRepulsiveLoss, self).__init__()
        self._vdw_table = _init_table()
        self.scale = 1.2
        self.vdw_cutoff = 4

    def vdw_table(self, device):
        if self._vdw_table.device != device:
            self._vdw_table = self._vdw_table.to(device)
        return self._vdw_table

    def get_masked_vdw_scale(self, mask):
        radii = self.vdw_table(mask.device)[:mask.shape[1]][mask[0]]
        return outer_sum(radii)

    def forward(self, output: Output, baseline: bool):
        """
        if baseline:
            aln = output.baseline_coords()
            pred_coords = aln.baseline_coords[aln.baseline_mask]
        else:
            aln = output.get_aligned_coords_n_masks(predicted=True, native=True)
            pred_coords = aln.predicted_coords[aln.predicted_mask]
        native_coords = aln.native_coords[aln.native_mask]

        coords = torch.cat((sample.bb_coords.detach(), coords), dim=-2)
        bb_coord_mask = torch.ones_like(bb_coords)[..., 0].bool()
        mask = torch.cat((bb_coord_mask, sample.sc_coord_mask),dim=-1)
        masked_coords = coords[mask]
        dists = torch.norm(to_rel_coord(masked_coords), dim=-1)
        # make sure diagonal is not counted towards loss
        dists = dists + (self.vdw_cutoff + 1) * torch.eye(dists.shape[0], device=dists.device)
        dev = (self.get_masked_vdw_scale(mask) / self.scale) - dists
        return torch.sum(torch.nn.ReLU()(dev) ** 2) / (dists[dists < self.vdw_cutoff].numel())
        """
        return 0
