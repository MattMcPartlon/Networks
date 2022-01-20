import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor
from typing import Tuple
from common.utils import exists
import torch
from common.helpers.neighbor_utils import NeighborInfo
from config.evoformer_config import EvoformerConfig
from evoformer.triangle_updates import PairUpdateLayer
from evoformer.node_updates import NodeUpdateLayer
from common.invariant.units import (
    Residual,
    LearnedOuterProd
)

List = nn.ModuleList  # noqa
Proj = lambda dim_in, dim_out: nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_out))


class Evoformer(nn.Module):
    def __init__(self, config: EvoformerConfig):
        super().__init__()
        self.config, self.layers = config, List([])
        node_in, node_hidden, node_out = config.node_dims
        edge_in, edge_hidden, edge_out = config.edge_dims

        # Input/Output projections
        self.node_project_in = Proj(node_in, node_hidden) if config.project_in else nn.Identity()
        self.edge_project_in = Proj(edge_in, edge_hidden) if config.project_in else nn.Identity()
        self.node_project_out = Proj(node_hidden, node_out) if config.project_out else nn.Identity()
        self.edge_project_out = Proj(edge_hidden, edge_out) if config.project_out else nn.Identity()
        self.layers = get_transformer_layers(config)

    def forward(
            self,
            node_feats: torch.Tensor,
            edge_feats: torch.Tensor,
            nbr_info: NeighborInfo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        config, device = self.config, node_feats.device
        mask = None
        if exists(nbr_info):
            nbr_info = nbr_info.to_device(device) if nbr_info.device != device else nbr_info
            mask = nbr_info.full_mask

        node_feats, edge_feats = self.project_in(node_feats=node_feats, edge_feats=edge_feats)

        for layer, (node_block, transition, transition_residual, pair_block) in enumerate(self.layers):
            # node to node attention (with edge bias)
            node_feats = node_block(nodes=node_feats, edges=edge_feats, nbr_info=nbr_info)
            # node to pair transition
            edge_feats = transition_residual(transition(node_feats), edge_feats)
            # triangle updates
            edge_feats = pair_block(edge_feats, mask=mask)

        return self.project_out(node_feats, edge_feats)

    def project_in(self, node_feats: Tensor, edge_feats: Tensor) -> Tuple[Tensor, Tensor]:
        return self.node_project_in(node_feats), self.edge_project_in(edge_feats)

    def project_out(self, node_feats: Tensor, edge_feats: Tensor) -> Tuple[Tensor, Tensor]:
        edge_feats = self.edge_project_out(edge_feats)
        if self.config.symmetrize_edges:
            edge_feats = (edge_feats + rearrange(edge_feats, "b i j d -> b j i d")) / 2
        return self.node_project_out(node_feats), edge_feats


def get_transformer_layers(config: EvoformerConfig):
    # set up transformer blocks
    edge_hidden, node_hidden = config.edge_dim_hidden, config.node_dim_hidden
    layers, edge_attn = List(), config.do_triangle_updates
    for i in range(config.depth):
        layers.append(List(
            [
                NodeUpdateLayer(
                    dim=node_hidden,
                    dim_head=config.node_dim_head,
                    heads=config.node_attn_heads,
                    edge_dim=edge_hidden,
                    use_rezero=config.use_rezero,
                    ff_mult=config.node_ff_mult,
                    dropout=config.node_dropout,
                ),
                LearnedOuterProd(
                    dim_in=node_hidden,
                    dim_out=edge_hidden,
                    dim_hidden=config.outer_prod_dim
                ),
                Residual(use_rezero=config.use_rezero),
                PairUpdateLayer(
                    dim=edge_hidden,
                    heads=config.edge_attn_heads,
                    dim_head=config.edge_dim_head,
                    dropout=config.edge_dropout,
                    tri_mul_dim=config.triangle_mul_dim,
                    do_checkpoint=config.checkpoint,
                    ff_mult=config.edge_ff_mult
                )
            ]
        ))

    return layers
