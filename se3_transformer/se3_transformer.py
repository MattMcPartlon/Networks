from typing import Union, Tuple, Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn
from abc import abstractmethod

from common.helpers.torch_utils import fused_gelu as GELU  # noqa
from common.helpers.neighbor_utils import NeighborInfo
from common.helpers.torch_utils import batched_index_select
from common.utils import exists
from config.se3_transformer_config import SE3TransformerConfig
from common.repr.basis import get_basis
from common.equivariant.attention.tfn_attention import TFNAttention

from common.equivariant.units.fiber_units import (
    FiberNorm,
    FiberFeedForwardResidualBlock,
    FiberFeedForward,
    FiberResidual,
    FiberDropout,
)


def get_attention_layer(config: SE3TransformerConfig) -> nn.Module:
    if config.attn_ty.lower() == 'tfn':
        return TFNAttention(
            fiber_in=config.fiber_hidden,
            config=config.attn_config(),
            tfn_config=config.tfn_config(),
            share_keys_and_values=config.share_keys_and_values,
        )
    raise Exception(f"Attention not implemented for: {config.attn_ty}")


class AttentionBlock(nn.Module):
    def __init__(
            self,
            config: SE3TransformerConfig,

    ):
        super().__init__()
        self.attn = get_attention_layer(config=config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.prenorm = FiberNorm(
            fiber=config.fiber_hidden,
            nonlin=config.nonlin,
        )
        self.residual = FiberResidual(use_re_zero=config.use_re_zero)
        self.dropout = FiberDropout(fiber=config.fiber_hidden, p=config.dropout)

    def forward(
            self,
            features: Dict[str, Tensor],
            edge_info: NeighborInfo,
            basis: Dict[str, Tensor],
            global_feats: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Attention Block

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features
        :param basis: equivariant basis mapping feats of type i to type j.
        :param global_feats: global features
        :return: dict mapping feature types to hidden features
        """
        res = features
        outputs = self.prenorm(features)
        outputs = self.attn(features=outputs, edge_info=edge_info, basis=basis,
                            global_feats=global_feats)
        return self.residual(self.dropout(outputs), res)


class AttentionLayer(nn.Module):
    def __init__(self,
                 config: SE3TransformerConfig
                 ):
        super().__init__()
        self.attn_block = AttentionBlock(config)

        self.ff_residual = FiberFeedForwardResidualBlock(
            feedforward=FiberFeedForward(
                fiber_in=config.fiber_hidden,
                hidden_mult=config.hidden_mult,
                n_hidden=1,
            ),
            pre_norm=FiberNorm(
                fiber=config.fiber_hidden,
                nonlin=config.nonlin,
            ),
            use_re_zero=config.use_re_zero,
        )

    def forward(self,
                features: Dict[str, torch.Tensor],
                edge_info: Tuple[torch.Tensor, NeighborInfo],
                basis,
                global_feats=None,
                ) -> Dict[str, torch.Tensor]:
        """Attention Layer

        norm(feats) -> x =AttentionBlock(feats) -> x = residual(x,feats)
        -> x = norm(x) -> residual(ff(x), x)

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features
        :param basis: equivariant basis mapping feats of type i to type j.
        :param global_feats: global features
        :return: dict mapping feature types to hidden features
        """
        attn_feats = self.attn_block(
            features=features,
            edge_info=edge_info,
            basis=basis,
            global_feats=global_feats,
        )
        return self.ff_residual(attn_feats)


class SE3Transformer(nn.Module):
    def __init__(
            self,
            config: SE3TransformerConfig
    ):
        super().__init__()
        self.config = config

        # global features
        self.accept_global_feats = exists(config.global_feats_dim)

        # Attention layers
        self.attn_layers = nn.ModuleList([
            AttentionLayer(config=config) for _ in range(config.depth)
        ])

    def forward(
            self,
            feats: Union[torch.Tensor, Dict[str, torch.Tensor]],
            edges: Optional[torch.Tensor] = None,
            neighbor_info: Optional[NeighborInfo] = None,
            global_feats: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """SE(3)-Equivariant Transformer

        :param feats: Either a tensor of shape (b,n,d_in) or a dict mapping feature type
        (e.g. "0" or "1") to a tensor of shape (b,n,d_0) for scalar feats and
        (b,n,d_1,3) for coord feats, where d_0 and d_1 are hidden dimensions of
        coords and scalars respectively.

        :param edges: Edge features of shape (b, n, n, d_e) where d_e is the edge dimension,
        or tensor of shape (b,n,N,d_e) where N is the number of neighbors to use per-point.
        (if tensor of shape (b,n,n,d_e) is passed, then the neighbors will be sub-selected
        based on NeighborInfo instance).

        :param neighbor_info: NeighborInfo instance

        :param global_feats: optional global features to use in each attention layer.

        :return: Dict of updated features.
        """
        config = self.config

        assert not (self.accept_global_feats ^ exists(
            global_feats)), 'you cannot pass in global features unless you init the class correctly'

        # convert features to dictionary representation
        feats = {'0': feats} if torch.is_tensor(feats) else feats
        feats['0'] = feats['0'] if len(feats['0'].shape) == 4 else feats['0'].unsqueeze(-1)
        global_feats = {'0': global_feats[..., None]} if torch.is_tensor(global_feats) else global_feats

        # check that input degrees and dimensions are as expected
        for deg, dim in config.fiber_in:
            feat_dim, feat_deg = feats[str(deg)].shape[-2:]
            assert dim == feat_dim, f" expected dim {dim} for input degree {deg}, got {feat_dim}"
            assert deg * 2 + 1 == feat_deg, f"wrong degree for feature {deg}, expected " \
                                            f": {deg * 2 + 1}, got : {feat_deg}"

        if exists(edges):
            if edges.shape[1] == edges.shape[2]:
                edges = batched_index_select(edges, neighbor_info.indices, dim=2)

        # get basis
        basis = self.compute_basis(
            neighbor_info=neighbor_info,
            max_degree=config.max_degrees - 1,
            differentiable=config.differentiable_coords
        )

        # main logic
        x, edge_info = feats, (edges, neighbor_info)

        x = self.project_in(features=x, edge_info=edge_info, basis=basis)

        for attn_layer in self.attn_layers:
            x = attn_layer(
                x,
                edge_info=edge_info,
                basis=basis,
                global_feats=global_feats,
            )

        return self.project_out(features=x, edge_info=edge_info, basis=basis)

    @staticmethod
    def compute_basis(neighbor_info: NeighborInfo, max_degree: int, differentiable: bool):
        basis = get_basis(neighbor_info.rel_pos.detach(),
                          max_degree=max_degree,
                          differentiable=differentiable
                          )
        # reshape basis for faster / more memory efficient kernel computation
        for key in basis:
            b = basis[key].shape[0]
            i, o = key.split(",")
            basis[key] = rearrange(basis[key], "... a b c d e -> ... a c d e b")
            n, top_k = neighbor_info.coords.shape[1], neighbor_info.top_k
            basis[key] = basis[key].reshape(b * n * top_k, 2 * int(i) + 1, -1)
        return basis

    @abstractmethod
    def project_in(
            self,
            features: Dict[str, Tensor],
            edge_info: Tuple[Optional[Tensor], NeighborInfo],
            basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Equivariant input projection

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features and neighbor info
        :param basis: equivariant basis mapping feats of type i to type j.
        :return: dict mapping feature types to hidden shapes
        """
        pass

    @abstractmethod
    def project_out(
            self,
            features: Dict[str, Tensor],
            edge_info: Tuple[Optional[Tensor], NeighborInfo],
            basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Equivariant output projection

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features and neighbor info
        :param basis: equivariant basis mapping feats of type i to type j.
        :return: dict mapping feature types to output shapes
        """
        pass
