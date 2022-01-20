import math
from enum import Enum
from functools import partial
from typing import Optional, Callable

import torch
from torch import nn, einsum

from common.utils import default
from common.invariant.units import FeedForward


class LinearInitTy(Enum):
    UNIFORM = 'uniform'
    NON_NEG_UNIFORM = "non_neg_uniform"
    DEFAULT = 'default'
    CONSTANT = 'constant'
    IDENTITY = 'identity'
    RELU = 'relu'


class LinearInit:
    def __init__(self,
                 weight_init_ty: LinearInitTy = LinearInitTy.DEFAULT,
                 weight_init_val: Optional[float] = None,
                 bias_init_ty: LinearInitTy = LinearInitTy.DEFAULT,
                 bias_init_val: Optional[float] = None,
                 use_bias: bool = False
                 ):
        self.weight_init_ty = weight_init_ty
        self.weight_init_val = weight_init_val
        self.bias_init_ty = bias_init_ty
        self.bias_init_val = bias_init_val
        self.use_bias = use_bias

    def init_func(self, override=False) -> Callable:
        return partial(
            _linear_init,
            wt_init_ty=self.weight_init_ty,
            wt_init_val=self.weight_init_val,
            bias_init_ty=self.bias_init_ty,
            bias_init_val=self.bias_init_val,
            override=override
        )


def _linear_init(
        param,
        wt_init_ty: LinearInitTy,
        bias_init_ty: LinearInitTy,
        wt_init_val: Optional[float] = None,
        bias_init_val: Optional[float] = None,
        override: bool = False,
):
    if wt_init_ty == LinearInitTy.IDENTITY:
        nn.init.eye_(param.weight)
        if param.bias is not None:
            nn.init.constant_(param.bias, 0)
        return

    if not override and (not isinstance(param, nn.Linear) or not isinstance(param, VNLinear)):
        return
    items = zip([wt_init_ty, bias_init_ty], ["weight", "bias"], [wt_init_val, bias_init_val])

    for (ty, key, val) in items:
        if not hasattr(param, key) or getattr(param, key) is None:
            continue

        if ty == LinearInitTy.DEFAULT:
            # from pytorch documentation
            # nn.init.kaiming_uniform_(getattr(param, key), a=math.sqrt(5))
            return

        elif ty == LinearInitTy.CONSTANT:
            assert wt_init_val is not None
            nn.init.constant_(getattr(param, key), val)

        elif ty == LinearInitTy.UNIFORM:
            nn.init.xavier_uniform_(getattr(param, key), gain=1.0)

        elif ty == LinearInitTy.RELU:
            nn.init.kaiming_uniform(getattr(param, key))

        elif ty == LinearInitTy.NON_NEG_UNIFORM:
            assert wt_init_val is not None
            nn.init.uniform_(getattr(param, key), a=0, b=val)

        else:
            raise Exception(f"could not find linear init ty {ty}")


class LinearKernel(nn.Module):

    def __init__(self, coord_dim, feature_dim, coord_dim_out=None, mult=2):
        super().__init__()
        coord_dim_out = default(coord_dim_out, coord_dim)
        self.transform = FeedForward(feature_dim, feature_dim * coord_dim_out * mult,
                                     coord_dim * coord_dim_out)
        self.coord_dim_out = coord_dim_out

    def forward(self, x, inv_feats):
        """
        Given atom features h_i and h_j, relative coordinate diffs d_ij, and edge
        features e_ij, learns a matrix W_ij to transform the relative coordinates
        (x_i-x_j).
        """
        d_in = x.shape[-2]
        view = inv_feats.shape[:-1]
        # can think of as a learned per-point kernel
        kernel = self.transform(inv_feats).reshape(*view, self.coord_dim_out, d_in)
        return einsum('...ij,...jk->...ik', kernel, x)


class VNLinear(nn.Module):
    """SE(k) equivariant Vector Neuron Linear Layer

    Any linear operator acting on points in R^{k} is SE(k) equivariant, since for any R in SO(k)
    W((v+t)R) = W(vR + tR) = (Wv)R+(Wt)R by associativity

    Note:
        we intentionally omit a bias term, as the addition of this term would  interfere with equivariance

    Reference:
        https://arxiv.org/pdf/2104.12229.pdf
        "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(
            self,
            dim_in: int,
            dim_out: int = None,
            init: LinearInit = None,
    ):
        super().__init__()
        init: LinearInit = default(init, LinearInit())
        self.weight, self.bias = nn.Parameter(torch.randn(dim_in, dim_out) / math.sqrt(dim_in)), None
        self.apply(init.init_func(override=True))

    def forward(self, x):
        return einsum('b n d m, d e -> b n e m', x, self.weight)
