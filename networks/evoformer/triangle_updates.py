import torch
from einops import rearrange
from torch import nn, einsum, Tensor
import torch.utils.checkpoint as checkpoint
from networks.common.utils import exists, default
from typing import Optional
from networks.common.invariant.units import Residual, FeedForward, PreNorm

max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa
List = nn.ModuleList  # noqa

tri_outgoing = lambda a, b: torch.einsum("b i k d, b j k d -> b i j d ", a, b)
tri_incoming = lambda a, b: torch.einsum("b k i d, b k j d -> b i j d ", a, b)


class TriangleMul(nn.Module):
    def __init__(self, dim_in, incoming: bool, dim_hidden=128):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim_in)
        self.to_feats_n_gates = nn.Linear(dim_in, 4 * dim_hidden)
        self.gate_out_proj = nn.Linear(dim_in, dim_in)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_in)
        )
        self.func = tri_outgoing if not incoming else tri_incoming

    def forward(self, edges, mask) -> Tensor:
        edges = self.pre_norm(edges)
        mask = ~mask.unsqueeze(-1)
        feats, gates = self.to_feats_n_gates(edges).chunk(2, -1)
        to_feats, from_feats = (feats * torch.sigmoid(gates)).chunk(2, -1)
        out_gate = torch.sigmoid(self.gate_out_proj(edges))
        to_feats, from_feats, out_gate = map(lambda x: x.masked_fill(mask, 0), (to_feats, from_feats, out_gate))
        return out_gate * self.to_out(self.func(to_feats, from_feats))


class TriangleAttn(nn.Module):
    def __init__(self, dim_in, starting: bool, dim_head=32, heads=4):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim_in)
        self.to_qkv = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.to_b = nn.Linear(dim_in, heads, bias=False)
        self.to_g = nn.Linear(dim_in, dim_head * heads)
        self.heads = heads
        self.scale = dim_head ** -(1 / 2)
        self.to_out = nn.Linear(dim_head * heads, dim_in)
        self.starting = starting

    def forward(self, edges, mask) -> Tensor:
        edges = self.pre_norm(edges)
        q, k, v = self.to_qkv(edges).chunk(3, -1)
        g = torch.sigmoid(self.to_g(edges))
        b = self.to_b(edges)
        q, k, v, b, g = map(lambda x: rearrange(x, "b n m (h d) -> b h n m d", h=self.heads),
                            (q, k, v, b, g))
        b = b.squeeze(-1)
        args = (q, k, b, v, g, mask)
        output = self.attn_starting(*args) if self.starting else self.attn_ending(*args)
        return self.to_out(rearrange(output, "b h i j d -> b i j (h d)"))

    def attn_starting(self, q, k, b, v, g, mask: Tensor):
        sim = torch.einsum("b h i j d, b h i k d -> b h i j k", q, k) * self.scale
        sim = sim + rearrange(b, "b h j k -> b h () j k")
        if exists(mask):
            attn_mask = ~rearrange(mask, "b i j -> b () i j ()")
            sim, g = map(lambda x: x.masked_fill(attn_mask, max_neg_value(sim)), (sim, g))
        attn = torch.softmax(sim, dim=-1)
        return g * einsum('...i j k,... i k d -> ... i j d', attn, v)

    def attn_ending(self, q, k, b, v, g, mask: Tensor):
        sim = torch.einsum("b h i j d, b h k j d -> b h i j k", q, k) * self.scale
        sim = sim + rearrange(b, "b h i k -> b h k () i")
        if exists(mask):
            attn_mask = ~rearrange(mask, "b i j -> b () i j ()")
            sim, g = map(lambda x: x.masked_fill(attn_mask, max_neg_value(sim)), (sim, g))
        attn = torch.softmax(sim, dim=-1)
        return g * einsum('...i j k,... k j d -> ... i j d', attn, v)


class PairUpdateLayer(nn.Module):
    def __init__(self,
                 dim,
                 use_rezero=True,
                 heads=4,
                 dim_head=24,
                 dropout=0,
                 tri_mul_dim: Optional[int] = None,
                 do_checkpoint: bool = False,
                 ff_mult: int = 2
                 ):
        super(PairUpdateLayer, self).__init__()
        Dropout = lambda rate: nn.Dropout(rate) if rate > 0 else nn.Identity()
        self.layers = List([
            List([
                TriangleMul(dim, incoming=False, dim_hidden=default(tri_mul_dim, dim)),
                Residual(use_rezero=use_rezero),
                Dropout(dropout)
            ]),
            List([
                TriangleMul(dim, incoming=True, dim_hidden=default(tri_mul_dim, dim)),
                Residual(use_rezero=use_rezero),
                Dropout(dropout)
            ]),
            List([
                TriangleAttn(dim_in=dim, dim_head=dim_head, heads=heads, starting=True),
                Residual(use_rezero=use_rezero),
                Dropout(dropout)
            ]),
            List([
                TriangleAttn(dim_in=dim, dim_head=dim_head, heads=heads, starting=False),
                Residual(use_rezero=use_rezero),
                Dropout(dropout)
            ]),
        ])
        self.transition = PreNorm(dim, FeedForward(dim, dim_hidden=ff_mult * dim))
        self.transition_residual = Residual(use_rezero=use_rezero)
        self.do_checkpoint = do_checkpoint

    def forward(self, edges: Tensor, mask: Tensor) -> Tensor:
        for net, residual, dropout in self.layers:
            out = checkpoint.checkpoint(net.forward, edges, mask) \
                if self.do_checkpoint else net(edges, mask)
            edges = residual(dropout(out), edges)
        return self.transition_residual(self.transition(edges), edges)
