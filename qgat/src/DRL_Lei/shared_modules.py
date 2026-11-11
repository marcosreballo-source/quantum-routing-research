"""
Shared encoder and attention modules for QGAT.

Contains the four classes duplicated between VRP_PPO_Model.py and VRP_Actor.py:
  GatConv, Encoder, Attention1, ProbAttention

Both files now import from here, eliminating code duplication.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math
import logging
from typing import Optional, Union

from DRL_Lei.quantum_layers import create_quantum_layer, SwitchableLinear

model_logger = logging.getLogger(__name__)


class GatConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int,
                 negative_slope: float = 0.2, dropout: Union[int, float] = 0,
                 use_quantum: bool = False,
                 use_quantum_attn: Optional[bool] = None,
                 n_qubits: int = 4, n_layers_quantum: int = 4,
                 rotation: str = 'RYRZ', topology: str = 'ring',
                 backend: str = 'lightning.qubit', diff_method: str = 'best',
                 init_weights: bool = True) -> None:
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.use_quantum = use_quantum
        # use_quantum_attn=None means "inherit from use_quantum"
        _use_quantum_attn = use_quantum if use_quantum_attn is None else use_quantum_attn

        self.fc = SwitchableLinear(
            in_channels, out_channels, use_quantum=use_quantum,
            n_qubits=n_qubits, n_layers=n_layers_quantum,
            rotation=rotation, topology=topology,
            backend=backend, diff_method=diff_method)
        self.attn = SwitchableLinear(
            2 * out_channels + edge_channels, out_channels,
            use_quantum=_use_quantum_attn, n_qubits=n_qubits, n_layers=n_layers_quantum,
            rotation=rotation, topology=topology,
            backend=backend, diff_method=diff_method)

        if use_quantum or _use_quantum_attn:
            model_logger.info(
                f"GatConv: fc={'quantum' if use_quantum else 'classical'}, "
                f"attn={'quantum' if _use_quantum_attn else 'classical'} "
                f"(n_qubits={n_qubits}, n_layers={n_layers_quantum})"
            )

        if init_weights:
            for name, p in self.named_parameters():
                if 'weight' in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, size: Optional[tuple] = None) -> torch.Tensor:
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)  # type: ignore[return-value]

    def message(self, edge_index_i: torch.Tensor, x_i: torch.Tensor,  # type: ignore[override]
                x_j: torch.Tensor, size_i: int, edge_attr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=float(self.dropout), training=self.training)
        return x_j * alpha

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                 conv_layers=3, n_heads=4,
                 quantum_fc_node=False, quantum_fc_edge=False, quantum_gat_layers=None,
                 quantum_gat_attn_layers=None,
                 n_qubits_actor=4, n_layers_quantum=4,
                 rotation='RYRZ', topology='ring',
                 backend='lightning.qubit', diff_method='best',
                 dropout_gat=0.0,
                 init_weights=True,
                 use_ffn=False, normalization='batch'):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.use_ffn = use_ffn
        self.normalization = normalization
        if quantum_gat_layers is None:
            quantum_gat_layers = []
        if quantum_gat_attn_layers is None:
            quantum_gat_attn_layers = []

        # Per-component quantum: fc_node
        self.fc_node = SwitchableLinear(
            input_node_dim, hidden_node_dim, use_quantum=quantum_fc_node,
            n_qubits=n_qubits_actor, n_layers=n_layers_quantum,
            rotation=rotation, topology=topology,
            backend=backend, diff_method=diff_method)
        if quantum_fc_node:
            model_logger.info("Encoder.fc_node: quantum")

        # Per-component quantum: fc_edge
        self.fc_edge = SwitchableLinear(
            input_edge_dim, hidden_edge_dim, use_quantum=quantum_fc_edge,
            n_qubits=n_qubits_actor, n_layers=n_layers_quantum,
            rotation=rotation, topology=topology,
            backend=backend, diff_method=diff_method)
        if quantum_fc_edge:
            model_logger.info("Encoder.fc_edge: quantum")

        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)

        # Per-layer quantum: each GAT layer enabled individually.
        # use_quantum controls the fc (node transform); use_quantum_attn controls attention.
        # When use_quantum_attn is None it inherits from use_quantum (backward-compatible).
        self.convs1 = nn.ModuleList([
            GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim,
                    dropout=dropout_gat,
                    use_quantum=(i in quantum_gat_layers),
                    use_quantum_attn=(i in quantum_gat_attn_layers) if quantum_gat_attn_layers else None,
                    n_qubits=n_qubits_actor, n_layers_quantum=n_layers_quantum,
                    rotation=rotation, topology=topology,
                    backend=backend, diff_method=diff_method,
                    init_weights=init_weights)
            for i in range(conv_layers)
        ])

        # Per-layer normalization after GAT residual
        def _make_norm(dim):
            if normalization == 'batch':
                return nn.BatchNorm1d(dim)
            elif normalization == 'layer':
                return nn.LayerNorm(dim)
            return nn.Identity()

        self.norms_attn = nn.ModuleList([_make_norm(hidden_node_dim) for _ in range(conv_layers)])

        # Optional FFN: Linear(d→4d) → ReLU → Linear(4d→d) with residual + norm
        if use_ffn:
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_node_dim, 4 * hidden_node_dim),
                    nn.ReLU(),
                    nn.Linear(4 * hidden_node_dim, hidden_node_dim),
                )
                for _ in range(conv_layers)
            ])
            self.norms_ffn = nn.ModuleList([_make_norm(hidden_node_dim) for _ in range(conv_layers)])

        if init_weights:
            for name, p in self.named_parameters():
                if 'weight' in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data.num_graphs
        x = torch.cat([data.x, data.demand], -1)
        x = self.fc_node(x)
        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)
        for i, conv in enumerate(self.convs1):
            x1 = conv(x, data.edge_index, edge_attr)
            x = x + x1  # residual
            x = self.norms_attn[i](x)
            if self.use_ffn:
                x2 = self.ffns[i](x)
                x = x + x2  # residual
                x = self.norms_ffn[i](x)
        x = x.reshape((batch_size, -1, self.hidden_node_dim))
        return x


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.0, dropout=0.0,
                 init_weights=True):
        super(Attention1, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if init_weights:
            for name, p in self.named_parameters():
                if 'weight' in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        '''
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: (batch_size,n_nodes,input_dim)
        :param mask: selected nodes  (batch_size,n_nodes)
        '''
        batch_size, n_nodes, input_dim = context.size()
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        compatibility = self.norm * torch.matmul(Q, K.transpose(2, 3))
        compatibility = compatibility.squeeze(2)
        compatibility = F.dropout(compatibility, p=self.attn_dropout, training=self.training)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(u_i, dim=-1)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)
        out_put = F.dropout(out_put, p=self.dropout, training=self.training)
        out_put = self.fc(out_put)
        return out_put


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim,
                 attn_dropout=0.0, dropout=0.0, temperature=2.5,
                 tanh_clipping=10.0,
                 init_weights=True):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = max(1e-6, float(temperature))
        self.tanh_clipping = tanh_clipping
        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim,
                                   attn_dropout=attn_dropout, dropout=dropout,
                                   init_weights=init_weights)
        if init_weights:
            for name, p in self.named_parameters():
                if 'weight' in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask, temperature=None):
        '''
        :param state_t: (batch_size,1,input_dim*3)
        :param context: (batch_size,n_nodes,input_dim)
        :param mask: (batch_size,n_nodes)
        :param temperature: override self.temperature if given
        '''
        temp = temperature if temperature is not None else self.temperature
        x = self.mhalayer(state_t, context, mask)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))
        compatibility = compatibility.squeeze(1)
        x = torch.tanh(compatibility)
        x = x * self.tanh_clipping
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x / temp, dim=-1)
        return scores

    def set_temperature(self, temperature):
        self.temperature = max(1e-6, float(temperature))
