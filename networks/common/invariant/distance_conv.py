import torch
from torch import nn
from typing import Tuple, Optional
from math import sqrt
from einops import rearrange
from networks.common.utils import exists
from torch import Tensor

to_rel_dist = lambda x: rearrange(x, "b n N h c -> b n N () h c") - \
                        rearrange(x, "b n N h c -> b n N h () c")
eps = lambda x: 10 * torch.finfo(x.dtype).eps  # noqa


def get_dists(dists: Optional[Tensor], coords: Optional[Tensor]):
    if exists(dists):
        return dists
    return torch.norm(to_rel_dist(coords).clamp_min(eps(coords)), dim=-1)


class PairwiseDistanceConv(nn.Module):

    def __init__(self, dim_in: Tuple[int, int], n_filters: int, pairwise=False, edge_dim=None):
        super().__init__()
        self.dim_in, self.n_filters = dim_in, n_filters
        dim_out = (dim_in[0] * (dim_in[1] - 1)) // 2
        if not pairwise:
            self.kernel = nn.Parameter(torch.randn(1, 1, 1, n_filters, dim_out) / sqrt(dim_in[0]))
            self.weights = nn.Parameter(torch.randn(1, 1, 1, n_filters, dim_out) / sqrt(dim_in[0]))
        else:
            assert edge_dim is not None
            self.kernel_and_weight_fn = nn.Sequential(
                nn.Linear(edge_dim, 4 * edge_dim),
                nn.GELU(),
                nn.Linear(4 * edge_dim, 2 * n_filters * dim_out, bias=True)
            )

        self.scale = 1 / sqrt(dim_in[0])
        self.pairwise = pairwise

    def forward(self, edges: Tensor, dists: Tensor) -> Tensor:
        if self.pairwise:
            KW = self.kernel_and_weight_fn(edges)
            K, W = map(
                lambda x: rearrange(
                    x, "... (f k) -> ... f k", f=self.n_filters),
                KW.chunk(2, dim=-1)
            )
        else:
            K, W = self.kernel, self.weights
        r, c = torch.triu_indices(*self.dim_in, 1)
        dists = dists[..., r, c]
        kds = nn.GELU()(1 - torch.square(dists.unsqueeze(-2) - K))
        weighted_kds = (kds * W) * self.scale
        return torch.sum(weighted_kds, dim=(-1))
