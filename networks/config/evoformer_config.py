from typing import Optional
from networks.common.utils import default


class EvoformerConfig:

    def __init__(
            self,
            node_dim_in: int,
            edge_dim_in: int,
            node_dim_hidden: Optional[int] = None,
            edge_dim_hidden: Optional[int] = None,
            node_dim_out: Optional[int] = None,
            edge_dim_out: Optional[int] = None,
            depth: int = 10,
            node_dropout: float = 0,
            edge_dropout: float = 0,
            edge_attn_heads: int = 4,
            edge_dim_head: int = 32,
            triangle_mul_dim: Optional[int] = None,
            outer_prod_dim: int = 16,
            node_attn_heads: int = 12,
            do_triangle_updates: bool = True,
            use_nbr_attn: bool = True,
            symmetrize_edges: bool = True,
            node_dim_head: int = 20,
            checkpoint: bool = True,
            project_in: bool = False,
            project_out: bool = True,
            node_ff_mult: int = 4,
            edge_ff_mult: int = 4,
            use_rezero: bool = True,

    ):
        self.node_dim_in, self.edge_dim_in = node_dim_in, edge_dim_in
        self.node_dim_hidden = default(node_dim_in, node_dim_hidden)
        self.node_dim_out = default(node_dim_out, node_dim_in)
        self.edge_dim_hidden = default(edge_dim_hidden, edge_dim_in)
        self.edge_dim_out = default(edge_dim_out, edge_dim_in)
        self.depth = depth
        self.node_dropout, self.edge_dropout = node_dropout, edge_dropout
        self.node_attn_heads, self.edge_attn_heads = node_attn_heads, edge_attn_heads
        self.node_dim_head, self.edge_dim_head = node_dim_head, edge_dim_head
        self.triangle_mul_dim = default(triangle_mul_dim, edge_dim_hidden)
        self.outer_prod_dim = outer_prod_dim
        self.do_triangle_updates = do_triangle_updates
        self.use_nbr_attn = use_nbr_attn
        self.symmetrize_edges = symmetrize_edges
        self.checkpoint = checkpoint
        self.project_in = project_in or self.node_dim_hidden != self.node_dim_in
        self.project_out = project_out or self.node_dim_hidden != self.node_dim_out
        self.node_ff_mult, self.edge_ff_mult = node_ff_mult, edge_ff_mult
        self.use_rezero = use_rezero

    @property
    def node_dims(self):
        return self.node_dim_in, self.node_dim_hidden, self.node_dim_out

    @property
    def edge_dims(self):
        return self.edge_dim_in, self.edge_dim_hidden, self.edge_dim_out
