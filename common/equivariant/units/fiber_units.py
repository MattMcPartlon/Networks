from typing import Dict, Optional

import torch
from torch import nn

from common.utils import default
from common.repr.fiber import Fiber
from common.equivariant.units.linear import VNLinear, LinearInit
from common.equivariant.units.norm import CoordNorm, SeqNormSE3
from common.constants import REZERO_INIT
from common.helpers.torch_utils import fused_gelu as GELU # noqa
NONLIN = GELU

from einops import rearrange


class FiberNorm(nn.Module):
    def __init__(self, fiber: Fiber,
                 nonlin: Optional[nn.Module] = NONLIN,
                 ty0_layernorm: bool = False,
                 seq_norm: bool = False
                 ):
        super().__init__()
        self.norms = nn.ModuleDict()
        nonlin = default(nonlin, nn.Identity)
        norm = CoordNorm if not seq_norm else SeqNormSE3
        for deg, dim in fiber:
            if Fiber.to_order(deg) == 1 and ty0_layernorm:
                self.norms[str(deg)] = nn.Sequential(nn.LayerNorm(dim), nonlin)
            else:
                self.norms[str(deg)] = norm(dim=dim, nonlin=nonlin)

    def forward(self, x):
        return {deg: self.norms[deg](feats) for deg, feats in x.items()}


class FiberResidual(nn.Module):
    """ only support instance where both Fibers are identical """

    def __init__(self, safe_mode=False, use_re_zero=True):
        super().__init__()
        self.safe_mode = safe_mode
        self.alpha = nn.Parameter(torch.zeros(2, requires_grad=True).float() + REZERO_INIT) if \
            use_re_zero else [1, 1]

    def forward(self, x: Dict[str, torch.Tensor], res):  # noqa
        out = {}
        for i, (degree, tensor) in enumerate(x.items()):
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = self.alpha[i] * out[degree] + res[degree]
            else:
                if self.safe_mode:
                    raise Exception("can't apply residual - fibers don't match!")
        return out


class FiberLinear(nn.Module):
    def __init__(
            self,
            fiber_in: Fiber,
            fiber_out: Fiber = None,
            init: Optional[LinearInit] = None
    ):
        super().__init__()
        init = default(init, LinearInit())
        fiber_out = default(fiber_out, fiber_in)
        self.projs = nn.ModuleDict()

        for (degree, dim_in, dim_out) in (fiber_in & fiber_out):
            self.projs[str(degree)] = VNLinear(
                dim_in=dim_in,
                dim_out=dim_out,
                init=init
            )

    def forward(self, x):
        return {deg: self.projs[deg](x[deg]) for deg in self.projs}


class FiberFeedForward(nn.Module):
    def __init__(
            self,
            fiber_in: Fiber,
            fiber_hidden: Optional[Fiber] = None,
            fiber_out: Optional[Fiber] = None,
            nonlin=NONLIN,
            hidden_mult: int = 1,
            n_hidden: int = 1,
            linear_init: LinearInit = None,
            seq_norm: bool = False
    ):

        super().__init__()
        fiber_out = default(fiber_out, fiber_in)
        self.layers = nn.ModuleList()
        hidden_fiber = default(fiber_hidden, fiber_in.scale(hidden_mult))
        layer_fibers = [fiber_in] + (n_hidden * [hidden_fiber]) + [fiber_out]

        for i, (f_in, f_out) in enumerate(zip(layer_fibers[:-1], layer_fibers[1:])):
            fiber_linear = FiberLinear(
                fiber_in=f_in,
                fiber_out=f_out,
                init=linear_init,
            )
            if i < len(layer_fibers) - 2:
                fiber_norm = FiberNorm(
                    fiber=f_out,
                    nonlin=nonlin,
                    ty0_layernorm=False,
                    seq_norm=seq_norm
                )
            else:
                fiber_norm = nn.Identity()
            self.layers.append(nn.Sequential(fiber_linear, fiber_norm))

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


class FiberFeedForwardResidualBlock(nn.Module):
    def __init__(self, feedforward: FiberFeedForward, pre_norm: nn.Module, use_re_zero=True):
        super().__init__()
        self.ff = feedforward
        self.residual = FiberResidual(use_re_zero=use_re_zero)
        self.pre_norm = pre_norm

    def forward(self, features):
        return self.residual(self.ff(self.pre_norm(features)), res=features)


class FiberDropout(nn.Module):
    def __init__(self, fiber: Fiber, p=0):
        super(FiberDropout, self).__init__()
        self.p = p
        self.dropout = nn.ModuleDict({str(d): nn.Dropout(p=p) if p > 0 else nn.Identity() for d in fiber.degrees})

    def forward(self, feats: Dict[str, torch.Tensor]):
        if self.p > 0:
            feats = {k: rearrange(v, "... d m -> ... m d") for k, v in feats.items()}
            feats = {k: self.dropout[k](v) for k, v in feats.items()}
            feats = {k: rearrange(v, "... m d -> ... d m") for k, v in feats.items()}
        return feats
