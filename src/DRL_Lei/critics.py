"""Critic architectures for Q-GAT VRP.

Five critic types are supported, selectable via ``quantum.critic.type``:

===================== ============================================================
Type                  Description
===================== ============================================================
shared_conv1d         3×Conv1d(k=1) 128→20→20→1, uses the actor's encoder output.
                      Default, lightweight (~2.6 k params).
                      (Lei 2020, also used in Giang 2024 Section 3.3)
shared_fc             3×Linear 128→20→20→1, uses the actor's encoder output.
                      Ablation variant of shared_conv1d.
own_encoder_conv1d    Own independent Q-GAT encoder + 3×Conv1d decoder.
                      Matches Giang 2024 Figure 1(a): critic has its own
                      Q-GAT encoder (φ) with independent weights, followed
                      by a 1D-Conv decoder.  Quantum-capable on both encoder
                      (via ``critic.encoder.*``) and decoder layers
                      (via ``critic.layers``).
own_encoder_mlp       Own independent Q-GAT encoder + MLP value head
                      (Linear(d,d) → ReLU → Linear(d,1)).
                      (Kool et al., Attention Learn to Route, 2019)
===================== ============================================================

Design Rationale
----------------
*  **shared_conv1d / shared_fc** share the actor's encoder output and apply a
   lightweight 3-layer decoder to estimate per-node values, summed to a scalar.
   This is parameter-efficient (~2.6 k params) and tightly couples actor &
   critic representations.  **Important:** these cannot be used as standalone
   ``baseline: critic`` in the rollout trainer because ``VRP_Actor`` does not
   expose encoder embeddings.
*  **own_encoder_conv1d** has its own full ``Encoder`` (PyG GAT, quantum-capable)
   followed by the same 3×Conv1d decoder as ``shared_conv1d``.  This matches
   the architecture diagram in Giang et al. (2024) Figure 1(a).
*  **own_encoder_mlp** has its own full ``Encoder`` plus a 2-layer MLP value
   head.  This follows Kool et al. (2019) and decouples actor/critic
   representations.

Backward Compatibility
----------------------
Old config names are automatically mapped via ``_LEGACY_NAME_MAP``:
``lei_conv1d`` / ``giang_conv1d`` → ``shared_conv1d``,
``legacy_fc`` → ``shared_fc``,
``kool_encoder`` → ``own_encoder_mlp``,
``conv1d`` / ``fc`` (old ``critic.decoder`` values) → ``shared_conv1d`` / ``shared_fc``.
"""

import logging

import torch.nn as nn
import torch.nn.functional as F

from DRL_Lei.shared_modules import Encoder
from DRL_Lei.VRP_PPO_Model import Critic as _CriticBase

model_logger = logging.getLogger(__name__)

# Re-export base critic under a canonical alias
CriticShared = _CriticBase

# ──────────────────────────────────────────────────────────────────────
# KOOL-style critic: own encoder + value head
# ──────────────────────────────────────────────────────────────────────

class CriticKOOL(nn.Module):
    """Independent-encoder critic following Kool et al. (2019).

    Architecture
    ~~~~~~~~~~~~
    1.  **Own Encoder** — identical architecture to the actor encoder
        (PyG GAT with optional quantum layers) but with *independent weights*.
        Quantum layers are configured via ``quantum.critic.encoder.*``,
        fully decoupled from the actor's ``quantum.encoder.*``.
    2.  **Mean-Pooling** — node embeddings → single graph embedding.
    3.  **Value Head** — ``Linear(d, d) → ReLU → Linear(d, 1)``.

    Why this design?
        * Decouples actor/critic learning dynamics (different LR, capacity).
        * Enables critic-only quantum ablation without touching the actor.
        * Mirrors the KOOL reference (``CriticNetwork`` + ``GraphAttentionEncoder``)
          but uses our PyG-based ``Encoder`` instead of vanilla MHA, preserving
          the full Q-GAT integration.

    Parameters
    ----------
    input_node_dim : int
        Raw node feature dimensionality (3 for CVRP: x, y, demand).
    hidden_node_dim : int
        Hidden dimension (must match actor for weight-sharing compatibility
        if ever needed).
    input_edge_dim : int
        Edge feature dimensionality.
    hidden_edge_dim : int
        Edge hidden dimension.
    conv_layers : int
        Number of GAT layers in the critic encoder.
    quantum_cfg : dict
        Top-level quantum config (inherits backend, diff_method, etc.).
    dropout_cfg : dict
        Dropout configuration.
    model_cfg : dict
        Model architecture config (use_ffn, normalization).
    """

    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim,
                 hidden_edge_dim, conv_layers=3,
                 quantum_cfg=None, dropout_cfg=None, model_cfg=None):
        super().__init__()
        if quantum_cfg is None:
            quantum_cfg = {}
        if dropout_cfg is None:
            dropout_cfg = {}
        if model_cfg is None:
            model_cfg = {}

        # Critic encoder quantum settings come from quantum.critic.encoder
        crit_enc = quantum_cfg.get('critic', {}).get('encoder', {})

        self.encoder = Encoder(
            input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
            conv_layers,
            quantum_fc_node=crit_enc.get('fc_node', False),
            quantum_fc_edge=crit_enc.get('fc_edge', False),
            quantum_gat_layers=crit_enc.get('gat_layers', []),
            quantum_gat_attn_layers=crit_enc.get('gat_attn_layers', []),
            n_qubits_actor=quantum_cfg.get('n_qubits_critic', 4),
            n_layers_quantum=quantum_cfg.get('n_layers_quantum_critic', 2),
            rotation=quantum_cfg.get('rotation', 'RYRZ'),
            topology=quantum_cfg.get('topology', 'ring'),
            backend=quantum_cfg.get('backend', 'lightning.qubit'),
            diff_method=quantum_cfg.get('diff_method', 'best'),
            dropout_gat=dropout_cfg.get('critic', 0.0),
            use_ffn=model_cfg.get('use_ffn', False),
            normalization=model_cfg.get('normalization', 'batch'),
        )

        # Value head: graph embedding → scalar (Kool et al., Section 4)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, 1),
        )

        model_logger.info(
            "CriticKOOL: own encoder (%d GAT layers, hidden=%d) + value_head",
            conv_layers, hidden_node_dim,
        )

    def forward(self, data):
        """Forward pass — receives raw PyG Data batch (NOT encoder output).

        Parameters
        ----------
        data : torch_geometric.data.Batch
            Batched PyG data with ``.x``, ``.edge_index``, ``.edge_attr``,
            ``.demand``.

        Returns
        -------
        value : Tensor, shape (batch_size,)
        """
        x = self.encoder(data)                       # (batch, nodes, hidden)
        graph_embedding = x.mean(dim=1)               # (batch, hidden)
        value = self.value_head(graph_embedding)       # (batch, 1)
        return value.squeeze(-1)                       # (batch,)


# ──────────────────────────────────────────────────────────────────────
# Own-encoder + Conv1d critic  (Giang 2024 Figure 1(a))
# ──────────────────────────────────────────────────────────────────────

class CriticOwnEncoderConv1d(nn.Module):
    """Independent-encoder critic with Conv1d decoder.

    Matches the architecture in Giang et al. (2024) Figure 1(a):
    the Critic Network has its **own** Q-GAT encoder (φ) with independent
    weights, followed by a 3×Conv1d(k=1) decoder (128→20→20→1).

    Architecture
    ~~~~~~~~~~~~
    1.  **Own Encoder** — identical architecture to the actor encoder
        (PyG GAT with optional quantum layers) but with *independent weights*.
        Quantum layers configured via ``quantum.critic.encoder.*``.
    2.  **Conv1d Decoder** — reuses the existing ``Critic`` class
        (3×Conv1d with optional per-layer quantum via ``critic.layers``).

    Parameters
    ----------
    input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim :
        Dimensions matching the actor encoder.
    conv_layers : int
        Number of GAT layers in the critic encoder.
    quantum_cfg : dict
        Top-level quantum config.
    quantum_critic_cfg : dict
        ``quantum.critic`` sub-section (layers, type, encoder).
    dropout_cfg : dict
        Dropout configuration.
    model_cfg : dict
        Model architecture config (use_ffn, normalization).
    """

    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim,
                 hidden_edge_dim, conv_layers=3,
                 quantum_cfg=None, quantum_critic_cfg=None,
                 dropout_cfg=None, model_cfg=None):
        super().__init__()
        if quantum_cfg is None:
            quantum_cfg = {}
        if quantum_critic_cfg is None:
            quantum_critic_cfg = {}
        if dropout_cfg is None:
            dropout_cfg = {}
        if model_cfg is None:
            model_cfg = {}

        # Critic encoder quantum settings come from quantum.critic.encoder
        crit_enc = quantum_cfg.get('critic', {}).get('encoder', {})

        self.encoder = Encoder(
            input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
            conv_layers,
            quantum_fc_node=crit_enc.get('fc_node', False),
            quantum_fc_edge=crit_enc.get('fc_edge', False),
            quantum_gat_layers=crit_enc.get('gat_layers', []),
            quantum_gat_attn_layers=crit_enc.get('gat_attn_layers', []),
            n_qubits_actor=quantum_cfg.get('n_qubits_critic', 4),
            n_layers_quantum=quantum_cfg.get('n_layers_quantum_critic', 2),
            rotation=quantum_cfg.get('rotation', 'RYRZ'),
            topology=quantum_cfg.get('topology', 'ring'),
            backend=quantum_cfg.get('backend', 'lightning.qubit'),
            diff_method=quantum_cfg.get('diff_method', 'best'),
            dropout_gat=dropout_cfg.get('critic', 0.0),
            use_ffn=model_cfg.get('use_ffn', False),
            normalization=model_cfg.get('normalization', 'batch'),
        )

        # Conv1d decoder (reuses existing Critic class)
        self.decoder = _CriticBase(
            hidden_node_dim,
            quantum_critic_cfg=quantum_critic_cfg,
            n_qubits_critic=quantum_cfg.get('n_qubits_critic', 4),
            n_layers_quantum_critic=quantum_cfg.get('n_layers_quantum_critic', 2),
            dropout_critic=dropout_cfg.get('critic', 0.0),
            rotation=quantum_cfg.get('rotation', 'RYRZ'),
            topology=quantum_cfg.get('topology', 'ring'),
            critic_decoder='conv1d',
            backend=quantum_cfg.get('backend', 'lightning.qubit'),
            diff_method=quantum_cfg.get('diff_method', 'best'),
        )

        model_logger.info(
            "CriticOwnEncoderConv1d: own encoder (%d GAT layers, hidden=%d) "
            "+ Conv1d decoder",
            conv_layers, hidden_node_dim,
        )

    def forward(self, data):
        """Forward pass — receives raw PyG Data batch (NOT encoder output).

        Parameters
        ----------
        data : torch_geometric.data.Batch
            Batched PyG data.

        Returns
        -------
        value : Tensor, shape (batch_size,)
        """
        x = self.encoder(data)       # (batch, nodes, hidden)
        return self.decoder(x)       # scalar per graph


# ──────────────────────────────────────────────────────────────────────
# Canonical type registry
# ──────────────────────────────────────────────────────────────────────

CRITIC_TYPES = ('shared_conv1d', 'shared_fc', 'own_encoder_conv1d', 'own_encoder_mlp')

# Backward-compat mapping: old names → canonical names
_LEGACY_NAME_MAP = {
    # Old type names
    'lei_conv1d': 'shared_conv1d',
    'giang_conv1d': 'shared_conv1d',
    'legacy_fc': 'shared_fc',
    'kool_encoder': 'own_encoder_mlp',
    # Old critic.decoder values
    'conv1d': 'shared_conv1d',
    'fc': 'shared_fc',
}


def _resolve_critic_type(critic_section: dict) -> str:
    """Resolve critic type from config, handling backward compatibility."""
    ctype = critic_section.get('type')
    if ctype is not None:
        return _LEGACY_NAME_MAP.get(ctype, ctype)
    # Fallback: old-style critic.decoder key
    decoder = critic_section.get('decoder', 'conv1d')
    return _LEGACY_NAME_MAP.get(decoder, 'shared_conv1d')


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def build_critic(critic_type, hidden_node_dim, *,
                 quantum_cfg=None, quantum_critic_cfg=None,
                 dropout_cfg=None, model_cfg=None,
                 input_node_dim=3, input_edge_dim=1,
                 hidden_edge_dim=16, conv_layers=3):
    """Build a critic ``nn.Module`` by type string.

    Parameters
    ----------
    critic_type : str
        One of ``CRITIC_TYPES``.
    hidden_node_dim : int
        Hidden dimension (128 by default).
    quantum_cfg : dict
        Top-level quantum config.
    quantum_critic_cfg : dict
        ``quantum.critic`` sub-section (layers list, type, encoder, …).
    dropout_cfg : dict
        Dropout config dict.
    model_cfg : dict
        Model config dict (use_ffn, normalization).
    input_node_dim : int
        Node input dim (for own-encoder critics).
    input_edge_dim : int
        Edge input dim (for own-encoder critics).
    hidden_edge_dim : int
        Edge hidden dim (for own-encoder critics).
    conv_layers : int
        Number of GAT layers (for own-encoder critics).

    Returns
    -------
    nn.Module
        Critic network.
    """
    if quantum_cfg is None:
        quantum_cfg = {}
    if quantum_critic_cfg is None:
        quantum_critic_cfg = {}
    if dropout_cfg is None:
        dropout_cfg = {}
    if model_cfg is None:
        model_cfg = {}

    # Backward compat: old names
    critic_type = _LEGACY_NAME_MAP.get(critic_type, critic_type)

    if critic_type == 'shared_conv1d':
        decoder_mode = 'conv1d'
    elif critic_type == 'shared_fc':
        decoder_mode = 'fc'
    elif critic_type == 'own_encoder_conv1d':
        return CriticOwnEncoderConv1d(
            input_node_dim, hidden_node_dim,
            input_edge_dim, hidden_edge_dim,
            conv_layers=conv_layers,
            quantum_cfg=quantum_cfg,
            quantum_critic_cfg=quantum_critic_cfg,
            dropout_cfg=dropout_cfg,
            model_cfg=model_cfg,
        )
    elif critic_type == 'own_encoder_mlp':
        return CriticKOOL(
            input_node_dim, hidden_node_dim,
            input_edge_dim, hidden_edge_dim,
            conv_layers=conv_layers,
            quantum_cfg=quantum_cfg,
            dropout_cfg=dropout_cfg,
            model_cfg=model_cfg,
        )
    else:
        raise ValueError(
            f"Unknown critic type: {critic_type!r}. "
            f"Choose from: {', '.join(CRITIC_TYPES)}"
        )

    # Shared-encoder critics — existing Critic with per-layer quantum
    return _CriticBase(
        hidden_node_dim,
        quantum_critic_cfg=quantum_critic_cfg,
        n_qubits_critic=quantum_cfg.get('n_qubits_critic', 4),
        n_layers_quantum_critic=quantum_cfg.get('n_layers_quantum_critic', 2),
        dropout_critic=dropout_cfg.get('critic', 0.0),
        rotation=quantum_cfg.get('rotation', 'RYRZ'),
        topology=quantum_cfg.get('topology', 'ring'),
        critic_decoder=decoder_mode,
        backend=quantum_cfg.get('backend', 'lightning.qubit'),
        diff_method=quantum_cfg.get('diff_method', 'best'),
    )
