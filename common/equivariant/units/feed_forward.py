import torch
from common.utils import default
from torch import nn

from common.equivariant.units.norm import CoordNorm
from math import sqrt
from common.helpers.torch_utils import fused_gelu as GELU  # noqa


class FeedForward(nn.Module):
    """SE(k) equivariant feed forward neural network

    This network consists of two linear projections, and works with
    all equivariant nonlinearities (see net_units.equivariant.nonlin).

    Reference:
        https://arxiv.org/pdf/2104.12229.pdf
        "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            dim_out=None,
            mult: int = 4,
            nonlin=None,
            use_norm=True,
            norm=None,
            init_eps=None,
    ):
        """SE(k) equivariant feedforward vector neural network.

        :param dim_in: input dimension
        :param dim_hidden: hidden dimension
        :param dim_out: output dimension
        :param mult: If fiber_hidden is not given, then the hidden dimensions
        for ech degree are taken as input_dimension * mult.
        :param nonlin: The non-linearity to use after projecting the input.
        to the hidden dimension(s).
        :param norm: Whether or not to apply normalization between
        projections. NOTE: if using PhaseNorm, layer norm is already built in.
        """
        super().__init__()
        dim_hidden = default(dim_hidden, dim_in * mult)
        dim_out = default(dim_out, dim_in)
        self.project_in = nn.Parameter(torch.randn(dim_hidden, dim_in) * default(init_eps, 1 / sqrt(dim_in)))
        self.nonlin = default(nonlin, GELU)
        self.project_out = nn.Parameter(torch.randn(dim_out, dim_hidden) * default(init_eps, 1 / sqrt(dim_hidden)))
        self.norm = default(norm, CoordNorm(dim_hidden)) if use_norm else nn.Identity()

    def forward(self, features):
        outputs = self.project_in.matmul(features)
        outputs = self.norm(outputs)
        outputs = self.nonlin(outputs)
        outputs = self.project_out.matmul(outputs)
        return outputs


class FeedForwardResidualBlock(nn.Module):
    """SE(k) equivariant feed-forward neural network block.

    (Optinally) Applies a norm to the input of the FeedForward network, and adds a residual connection.

    (1) out = norm_1(x) # learned equivariant norm based non-linearity --optional
    (2) out = linear_out(out) # learned linear projection -> (degree, dim_in)->(degree, dim_hidden)
    (3) out = norm_2(out) # learned equivariant norm based non-linearity
    (4) out = linear_out(out) #learned linear projection # (degree, dim_hidden) -> (degree, dim_in)
    (5) out = residual(x, out) #residual connection
    """

    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            mult: int = 4,
            nonlin=None,
            use_norm=True,
            norm=None,
            pre_norm=True,

    ):
        """Vector Neuron Feedforward Block with residual connection.

        :param dim_in: input dimension
        :param dim_hidden: hidden dimension
        :param mult: If dim_hidden is not given, then the hidden dimensions
        are taken as input_dimension * mult.
        :param nonlin: The non-linearity to use after projecting the input.
        to the hidden dimension.
        :param norm: whether or not to apply layer-norm on the inputs
        for each degree.
        :param pre_norm: whether or not to apply a norm based non-linearity to
        the input before passing through a feedforward layer.
        """
        super().__init__()
        self.prenorm = CoordNorm(dim_in) if pre_norm else lambda x: x
        self.feedforward = FeedForward(dim_in,
                                       dim_hidden=dim_hidden,
                                       dim_out=dim_in,
                                       mult=mult,
                                       nonlin=nonlin,
                                       norm=norm,
                                       use_norm=use_norm)

    def forward(self, features):
        res = features
        return self.feedforward(self.prenorm(features)) + res
