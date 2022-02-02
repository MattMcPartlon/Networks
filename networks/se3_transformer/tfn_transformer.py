from typing import Tuple, Dict, Optional

from torch import Tensor
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn

from networks.common.helpers.torch_utils import fused_gelu as GELU  # noqa
from networks.common.helpers.neighbor_utils import NeighborInfo
from networks.common.utils import exists
from networks.config.se3_transformer_config import SE3TransformerConfig
from networks.common.equivariant.tfn.tfn import ConvSE3
from networks.se3_transformer.se3_transformer import SE3Transformer

from networks.common.equivariant.units.fiber_units import (
    FiberNorm,
    FiberResidual,
    FiberLinear,
)


class TFNTransformer(SE3Transformer):
    def __init__(
            self,
            config: SE3TransformerConfig
    ):
        super().__init__(config=config)

        # pre and post TFN layers for mapping between feature types
        conv_config = config.tfn_config(self_interaction=True, pool=True)
        conv_in_config = lambda fiber_in: conv_config.override(fiber_in=fiber_in, fiber_out=config.fiber_hidden)
        conv_out_config = lambda fiber_out: conv_config.override(fiber_in=config.fiber_hidden, fiber_out=fiber_out)

        # convolution layers before and after attention
        self.pre_tfn_layers, self.post_tfn_layers = nn.ModuleList(), nn.ModuleList()
        self.conv_in = ConvSE3(conv_in_config(fiber_in=config.fiber_in))
        self.conv_in_pre_norm = FiberNorm(config.fiber_in, nonlin=nn.Identity())

        self.conv_out_pre_norm, self.conv_out = None, None
        if config.conv_out_layers > 0:
            self.conv_out_pre_norm = FiberNorm(fiber=config.fiber_hidden, nonlin=config.nonlin)
            self.conv_out = ConvSE3(
                conv_out_config(fiber_out=config.fiber_hidden if config.project_out else config.fiber_out)
            )

        for _ in range(max(0, config.conv_in_layers - 1)):
            self.pre_tfn_layers.append(
                nn.ModuleList([
                    ConvSE3(conv_in_config(fiber_in=config.fiber_in)),
                    FiberNorm(fiber=config.fiber_hidden, nonlin=config.nonlin),
                    FiberResidual(use_re_zero=config.use_re_zero)
                ])
            )
        for _ in range(max(0, config.conv_out_layers - 1)):
            self.post_tfn_layers.append(
                nn.ModuleList([
                    ConvSE3(conv_out_config(fiber_out=config.fiber_hidden)),
                    FiberNorm(fiber=config.fiber_hidden, nonlin=config.nonlin),
                    FiberResidual(use_re_zero=config.use_re_zero)
                ])
            )

        self.out_proj = None
        if config.project_out:
            self.out_proj = nn.Sequential(
                FiberNorm(config.fiber_hidden, nonlin=nn.Identity()),
                FiberLinear(config.fiber_hidden, config.fiber_out)
            )

    def project_in(
            self,
            features: Dict[str, Tensor],
            edge_info: Tuple[Optional[Tensor], NeighborInfo],
            basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # project in -- pre norm in attention layer will handle conv output
        x = features
        x = self.conv_in(self.conv_in_pre_norm(x), edge_info, basis=basis)
        for conv, nonlin, residual in self.pre_tfn_layers:
            res = x
            out = conv(nonlin(x), edge_info=edge_info, basis=basis)
            x = residual(out, res=res)
        return x

    def project_out(
            self,
            features: Dict[str, Tensor],
            edge_info: Tuple[Optional[Tensor], NeighborInfo],
            basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # project out
        x = features
        if exists(self.conv_out):
            x = self.conv_out(self.conv_out_pre_norm(x), edge_info=edge_info, basis=basis)
        for i, (conv, nonlin, residual) in enumerate(self.post_tfn_layers):
            res = x
            x = nonlin(x)
            x = residual(conv(x, edge_info, basis=basis), res)
        if exists(self.out_proj):
            x = self.out_proj(x)

        return x
