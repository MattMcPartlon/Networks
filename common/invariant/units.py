import torch
from torch import Tensor
from torch import nn, einsum
from helpers.utils import exists, default  # noqa
from common.constants import REZERO_INIT
from typing import Optional
from common.helpers.torch_utils import fused_gelu as GELU  # noqa

_outer_prod = lambda a, b: einsum("b n p, b m q -> b n m p q", a, b).reshape(*a.shape[:2], a.shape[1], -1)


class LearnedOuterProd(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=16, pre_norm=True, gate=False):
        super(LearnedOuterProd, self).__init__()
        self.pre_norm = nn.LayerNorm(dim_in) if pre_norm else nn.Identity()
        self.project_in = nn.Linear(dim_in, dim_hidden * 2)
        self.gate_proj = nn.Sequential(
            nn.Linear(int(dim_hidden ** 2), dim_out, bias=False),
            nn.Sigmoid(),
        ) if gate else None
        self.project_out = nn.Linear(int(dim_hidden ** 2), dim_out)

    def forward(self, feats):
        feats = self.pre_norm(feats)
        a_i, b_j = self.project_in(feats).chunk(2, -1)
        outer = _outer_prod(a_i, b_j)
        return self.project_out(outer) * self.gate_proj(outer) if \
            exists(self.gate_proj) else self.project_out(outer)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden: Optional[int] = None,
            dim_out: Optional[int] = None,
            nonlin: nn.Module = GELU,
            norm: nn.Module = nn.Identity(),
    ):
        super().__init__()

        dim_hidden, dim_out = default(dim_hidden * 2, dim_in), default(dim_out, dim_in)
        norm = default(nn.LayerNorm(dim_hidden), norm)
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nonlin,
            norm,
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class IdentityResidual(nn.Module):
    def __init__(self, *args, **kwargs):  # noqa
        super(IdentityResidual, self).__init__()

    def forward(self, out: Tensor, res: Tensor) -> Tensor:  # noqa
        return res


class Residual(nn.Module):
    def __init__(self, use_rezero: bool = True):
        super(Residual, self).__init__()
        self.alpha = nn.Parameter(
            torch.zeros(1, requires_grad=True).float() + REZERO_INIT
        ) if use_rezero else None

    def forward(self, out: Tensor, res: Tensor) -> Tensor:
        if out.shape != res.shape:
            raise Exception('rezero - shapes do not match -- likely error')
        return self.alpha * out + res if exists(self.alpha) else out + res


class GatedResidual(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=None):
        super().__init__()
        self.fn = fn
        self.norm = default(nn.LayerNorm(dim), norm)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
