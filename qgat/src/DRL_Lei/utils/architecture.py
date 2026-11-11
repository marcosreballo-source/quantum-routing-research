"""
Architecture description and strategy identification for quantum ablation studies.

Prints a clear visual of which model components are quantum [Q] vs classical [C],
and identifies the ablation strategy (A/B/C/D/E) from config.
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_experiment_id(cfg, config_path=None) -> str:
    """Extract experiment_id from config or infer from config file path.

    Priority:
      1. cfg.experiment_id (explicit field in YAML)
      2. Infer from config_path filename (e.g., phase2a/strategy_A_encoder_only.yaml -> phase2a_strategyA)
      3. Fallback: identify_strategy() slug
    """
    if hasattr(cfg, '__dict__'):
        cfg_dict = _ns_to_dict(cfg)
    else:
        cfg_dict = cfg

    # 1. Explicit field
    eid = cfg_dict.get('experiment_id')
    if eid:
        return str(eid)

    # 2. Infer from config path
    if config_path is not None:
        p = Path(config_path)
        parent = p.parent.name  # e.g. "phase2a", "ablations"
        stem = p.stem           # e.g. "strategy_A_encoder_only"
        if parent and parent != 'configs':
            return f"{parent}_{stem}"
        return stem

    # 3. Fallback: slugify the strategy name
    strategy = identify_strategy(cfg_dict)
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', strategy).strip('_').lower()
    return slug


def describe_architecture(cfg) -> str:
    """Generate human-readable architecture description from config namespace or dict."""
    if hasattr(cfg, '__dict__'):
        cfg = _ns_to_dict(cfg)

    q = cfg.get('quantum', {})
    if not q.get('enabled', False):
        return (
            "=" * 70 + "\n"
            " ARCHITECTURE: Classical Model (No Quantum)\n"
            + "=" * 70
        )

    rotation = q.get('rotation', 'RYRZ')
    topology = q.get('topology', 'ring')
    nq = q.get('n_qubits_actor', 4)
    nl = q.get('n_layers_quantum_actor', 4)
    conv_layers = cfg.get('model', {}).get('conv_layers', 4)

    enc = q.get('encoder', {})
    dec = q.get('decoder', {})
    crt = q.get('critic', {})

    # GAT has two sub-components: fc (feature transform) and attn (attention mechanism)
    gat_fc_q = set(enc.get('gat_layers', []))
    gat_attn_q = set(enc.get('gat_attn_layers', []))
    fc_node_q = enc.get('fc_node', False)
    fc_edge_q = enc.get('fc_edge', False)
    dec_fc = dec.get('fc', False)
    dec_fc1 = dec.get('fc1', False)
    crt_q = bool(crt.get('layers', []))

    strategy = identify_strategy(cfg)

    lines = [
        "=" * 70,
        f"  ARCHITECTURE: {strategy}",
        f"  Backend: {q.get('backend', '?')} | Diff Method: {q.get('diff_method', '?')} | "
        f"Rotation: {rotation} | Topology: {topology} | Qubits: {nq} | Depth: {nl}",
        "=" * 70,
        "",
        "  ENCODER (Graph Attention Network):",
        f"    fc_node: {'[Q]' if fc_node_q else '[C]'}   "
        f"fc_edge: {'[Q]' if fc_edge_q else '[C]'}",
    ]

    # Show GAT layer structure with fc and attn separately
    if gat_fc_q or gat_attn_q:
        lines.append("    GAT layers (fc component):   " + "  ".join(
            f"L{i}{'[Q]' if i in gat_fc_q else '[C]'}"
            for i in range(conv_layers)
        ))
        lines.append("    GAT layers (attn component): " + "  ".join(
            f"L{i}{'[Q]' if i in gat_attn_q else '[C]'}"
            for i in range(conv_layers)
        ))
    else:
        lines.append("    GAT layers: " + "  ".join(
            f"L{i}[C]"
            for i in range(conv_layers)
        ))

    lines.extend([
        "",
        "  ACTOR DECODER (Pointer Network):",
        f"    context_proj (fc):  {'[Q]' if dec_fc else '[C]'}",
        f"    node_proj    (fc1): {'[Q]' if dec_fc1 else '[C]'}",
        "",
        "  CRITIC NETWORK:",
        f"    type: {crt.get('type', crt.get('decoder', 'shared_conv1d'))}",
        f"    quantum layers: {'[Q] ' + str(list(crt.get('layers', []))) if crt_q else '[C] (all classical)'}",
        "",
        "-" * 70,
    ])

    # Summary counts: encoder is quantum if either fc or attn has quantum
    encoder_has_quantum = bool(gat_fc_q or gat_attn_q)
    n_q = sum([
        encoder_has_quantum,
        dec_fc or dec_fc1,
        crt_q,
    ])
    lines.append(f"  Quantum components: {n_q}/3  "
                 f"(encoder={'Q' if encoder_has_quantum else 'C'}, "
                 f"actor_dec={'Q' if (dec_fc or dec_fc1) else 'C'}, "
                 f"critic={'Q' if crt_q else 'C'})")
    lines.append("=" * 70)

    return "\n".join(lines)


def identify_strategy(cfg) -> str:
    """Identify Phase-2a ablation strategy from config."""
    if hasattr(cfg, '__dict__'):
        cfg = _ns_to_dict(cfg)

    q = cfg.get('quantum', {})
    if not q.get('enabled', False):
        return "Classical (no quantum)"

    enc = q.get('encoder', {})
    dec = q.get('decoder', {})
    crt = q.get('critic', {})

    # Consider both gat_layers and gat_attn_layers for encoder quantum status
    gat_q = enc.get('gat_layers', []) or enc.get('gat_attn_layers', [])
    dec_q = dec.get('fc', False) or dec.get('fc1', False)
    crt_q = bool(crt.get('layers', []))

    conv_layers = cfg.get('model', {}).get('conv_layers', 4)
    encoder_partial = 0 < (len(enc.get('gat_layers', [])) + len(enc.get('gat_attn_layers', []))) < (2 * conv_layers)

    if encoder_partial and not dec_q and not crt_q:
        return f"E — Partial Encoder (gat_layers={enc.get('gat_layers', [])} + gat_attn_layers={enc.get('gat_attn_layers', [])})"
    if dec_q and crt_q:
        return "D — Full Quantum (encoder+actor_dec+critic)"
    if dec_q:
        return "B — Encoder + Actor Decoder Quantum"
    if crt_q:
        return "C — Encoder + Critic Quantum"
    if gat_q:
        return "A — Encoder Only (Paper Giang)"
    return "Unknown"


def _ns_to_dict(obj):
    """Recursively convert namespace to dict."""
    if hasattr(obj, '__dict__'):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(i) for i in obj]
    return obj
