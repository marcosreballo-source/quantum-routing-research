"""
Generate all ablation YAML configuration files for the QGAT vs Classical DRL study.

Strategy
--------
Phase 1A (Light classical sweep) — 3 seeds × {Lei/Kool} × {4+2 critic types} × {PPO/Rollout}
Phase 1B (HP sensitivity on best config) — 3 seeds × lr/bs/entropy/data_size variations
Phase 1C (Scale validation) — 5 seeds × top-2 configs × n_nodes {20,50,100}
Phase 2  (Quantum variants) — 3 seeds × quantum strategies + Giang paper-exact replica
Phase 3  (Optional quantum circuit tuning) — 3 seeds × ansatz / topology variations

Usage
-----
    python scripts/generate_ablation_yamls.py                   # all phases
    python scripts/generate_ablation_yamls.py --phases 1A 1B    # selected phases
    python scripts/generate_ablation_yamls.py --dry-run         # print only, no writes

Output: ablation_yamls/{phase}/*.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml  # pyyaml

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # baseline_merged/
OUTPUT_ROOT = BASE_DIR / "ablation_yamls"


def _write(path: Path, content: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write: {path.relative_to(BASE_DIR)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  wrote: {path.relative_to(BASE_DIR)}")


def _mode_block() -> dict:
    """All ablation YAML runs use hybrid_gpu mode (SageMaker GPU instances)."""
    return {"mode": "hybrid_gpu"}


def _seeds(data_seed: int, model_seed: int) -> dict:
    return {"seeds": {"data_seed": data_seed, "model_seed": model_seed}}


SEEDS_3 = [
    {"data_seed": 42, "model_seed": 42},
    {"data_seed": 123, "model_seed": 123},
    {"data_seed": 456, "model_seed": 456},
]
SEEDS_5 = SEEDS_3 + [
    {"data_seed": 789, "model_seed": 789},
    {"data_seed": 999, "model_seed": 999},
]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1A — Light classical sweep
# ─────────────────────────────────────────────────────────────────────────────
# Model × Critic × Method × Seed
#
# Lei model  = GatConv encoder + ProbAttention decoder (edge features)
#   critic options: shared_conv1d (default), shared_fc, own_encoder_conv1d, own_encoder_mlp
# Kool model = Attention1 encoder + ProbAttention decoder (NO edge features)
#   critic options: own_encoder_mlp (canonical Kool), shared_conv1d (ablation)
#
# Data sizes: 10k, 50k, 100k  (to validate data efficiency)
# n_nodes: 20
# ─────────────────────────────────────────────────────────────────────────────

LEI_CRITIC_TYPES = ["shared_conv1d", "shared_fc", "own_encoder_conv1d", "own_encoder_mlp"]
KOOL_CRITIC_TYPES = ["own_encoder_mlp", "shared_conv1d"]

# Kool uses Attention1 encoder — modelled via model.use_attention1 = true
# (When use_attention1 is true the actor ignores edge features.)
# Lei is the default (GatConv).

DATA_SIZES_1A = [10_000, 50_000, 100_000]


def _ppo_training_block(data_size: int) -> dict:
    return {
        "training": {
            "method": "ppo",
            "policy_mode": "sampling",
            "lr": 3.0e-4,
            "batch_size": 512,
            "epochs": 100,
            "timestep": 1,
            "ppo_epoch": 3,
            "entropy_value": 0.01,
            "eps_clip": 0.2,
        },
        "dataset": {"data_size": data_size, "val_size": 10_000},
    }


def _rollout_training_block(data_size: int) -> dict:
    return {
        "training": {
            "method": "rollout",
            "lr": 1.0e-3,
            "batch_size": 512,
            "epochs": 100,
            "lr_decay": 0.96,
        },
        "dataset": {"data_size": data_size, "val_size": 10_000},
    }


def _critic_block(critic_type: str) -> dict:
    """Build the quantum section needed to set critic.type (classical critic selection)."""
    return {
        "quantum": {
            "enabled": False,
            "critic": {
                "type": critic_type,
            },
        }
    }


def _lei_model_block() -> dict:
    return {
        "problem": {"n_nodes": 20},
        "model": {
            "hidden_node_dim": 128,
            "hidden_edge_dim": 16,
            "conv_layers": 4,
            "use_ffn": False,
            "normalization": "batch",
        },
    }


def _kool_model_block() -> dict:
    """
    Kool 2019 architecture:
      - Attention1 MHA encoder (no edge features) activated by use_ffn=True + normalization=layer
        and the special flag model.use_attention1=true
      - Same ProbAttention decoder
    """
    return {
        "problem": {"n_nodes": 20},
        "model": {
            "hidden_node_dim": 128,
            "hidden_edge_dim": 16,
            "conv_layers": 4,
            "use_ffn": True,        # Kool-style FFN after each attention layer
            "normalization": "layer",  # Kool uses layer norm
            "use_attention1": True,    # flag to swap GatConv → Attention1
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def phase1a(dry_run: bool) -> None:
    print("\n[Phase 1A] Lei × Kool classical sweep …")
    phase_dir = OUTPUT_ROOT / "phase1A"
    count = 0

    for data_size in DATA_SIZES_1A:
        ds_tag = f"{data_size // 1000}k"

        # ── Lei model ────────────────────────────────────────────────────────
        for critic in LEI_CRITIC_TYPES:
            for method in ("ppo", "rollout"):
                for seed in SEEDS_3:
                    s = seed["data_seed"]
                    eid = f"1A_lei_{critic}_{method}_ds{ds_tag}_seed{s}"
                    train = _ppo_training_block(data_size) if method == "ppo" else _rollout_training_block(data_size)
                    cfg: dict[str, Any] = {"experiment_id": eid}
                    cfg = _deep_merge(cfg, _mode_block())
                    cfg = _deep_merge(cfg, _lei_model_block())
                    cfg = _deep_merge(cfg, train)
                    cfg = _deep_merge(cfg, _critic_block(critic))
                    cfg = _deep_merge(cfg, {"seeds": seed})
                    fname = f"lei_{critic}_{method}_ds{ds_tag}_seed{s}.yaml"
                    _write(phase_dir / fname, cfg, dry_run)
                    count += 1

        # ── Kool model ───────────────────────────────────────────────────────
        for critic in KOOL_CRITIC_TYPES:
            for method in ("ppo", "rollout"):
                for seed in SEEDS_3:
                    s = seed["data_seed"]
                    eid = f"1A_kool_{critic}_{method}_ds{ds_tag}_seed{s}"
                    train = _ppo_training_block(data_size) if method == "ppo" else _rollout_training_block(data_size)
                    cfg = {"experiment_id": eid}
                    cfg = _deep_merge(cfg, _mode_block())
                    cfg = _deep_merge(cfg, _kool_model_block())
                    cfg = _deep_merge(cfg, train)
                    cfg = _deep_merge(cfg, _critic_block(critic))
                    cfg = _deep_merge(cfg, {"seeds": seed})
                    fname = f"kool_{critic}_{method}_ds{ds_tag}_seed{s}.yaml"
                    _write(phase_dir / fname, cfg, dry_run)
                    count += 1

    print(f"  Phase 1A: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1B — HP sensitivity (uses best config from 1A, hardcoded defaults)
# ─────────────────────────────────────────────────────────────────────────────
# Best config assumption (placeholder): Lei + shared_conv1d + PPO + 100k data
# Update `BEST_1A_BASE` after Phase 1A evaluation.
# ─────────────────────────────────────────────────────────────────────────────

BEST_1A_BASE = {
    "model": {
        "hidden_node_dim": 128,
        "hidden_edge_dim": 16,
        "conv_layers": 4,
        "use_ffn": False,
        "normalization": "batch",
    },
    "quantum": {
        "enabled": False,
        "critic": {"type": "shared_conv1d"},
    },
}

HP_SWEEPS = {
    "lr": [1.0e-4, 3.0e-4, 1.0e-3],
    "batch_size": [256, 512, 1024],
    "entropy_value": [0.001, 0.01, 0.05],
    "data_size": [50_000, 100_000, 512_000],
}


def phase1b(dry_run: bool) -> None:
    print("\n[Phase 1B] HP sensitivity sweep …")
    phase_dir = OUTPUT_ROOT / "phase1B"
    count = 0

    for hp_name, hp_values in HP_SWEEPS.items():
        for val in hp_values:
            for seed in SEEDS_3:
                s = seed["data_seed"]
                val_tag = str(val).replace(".", "p").replace("e-", "e-").replace("0000", "")
                eid = f"1B_{hp_name}_{val_tag}_seed{s}"
                cfg: dict[str, Any] = {"experiment_id": eid}
                cfg = _deep_merge(cfg, _mode_block())
                cfg = _deep_merge(cfg, {"problem": {"n_nodes": 20}})
                cfg = _deep_merge(cfg, BEST_1A_BASE)
                # Build training override
                if hp_name == "data_size":
                    train_override = {
                        "training": {
                            "method": "ppo",
                            "policy_mode": "sampling",
                            "lr": 3.0e-4,
                            "batch_size": 512,
                            "epochs": 100,
                            "timestep": 1,
                            "ppo_epoch": 3,
                            "entropy_value": 0.01,
                            "eps_clip": 0.2,
                        },
                        "dataset": {"data_size": int(val), "val_size": 10_000},
                    }
                else:
                    train_base = {
                        "training": {
                            "method": "ppo",
                            "policy_mode": "sampling",
                            "lr": 3.0e-4,
                            "batch_size": 512,
                            "epochs": 100,
                            "timestep": 1,
                            "ppo_epoch": 3,
                            "entropy_value": 0.01,
                            "eps_clip": 0.2,
                        },
                        "dataset": {"data_size": 100_000, "val_size": 10_000},
                    }
                    train_base["training"][hp_name] = val
                    train_override = train_base
                cfg = _deep_merge(cfg, train_override)
                cfg = _deep_merge(cfg, {"seeds": seed})
                fname = f"{hp_name}_{val_tag}_seed{s}.yaml"
                _write(phase_dir / fname, cfg, dry_run)
                count += 1

    print(f"  Phase 1B: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1C — Scale validation (n_nodes: 20, 50, 100)
# Top-2 configs from 1A (placeholder: Lei/shared_conv1d/ppo + Kool/own_encoder_mlp/ppo)
# 5 seeds
# ─────────────────────────────────────────────────────────────────────────────

TOP2_CONFIGS = [
    {
        "tag": "lei_shared_conv1d_ppo",
        "model": _lei_model_block()["model"],
        "training_method": "ppo",
        "critic": "shared_conv1d",
    },
    {
        "tag": "kool_own_encoder_mlp_ppo",
        "model": _kool_model_block()["model"],
        "training_method": "ppo",
        "critic": "own_encoder_mlp",
    },
]
N_NODES_1C = [20, 50, 100]


def phase1c(dry_run: bool) -> None:
    print("\n[Phase 1C] Scale validation …")
    phase_dir = OUTPUT_ROOT / "phase1C"
    count = 0

    for config in TOP2_CONFIGS:
        for n_nodes in N_NODES_1C:
            for seed in SEEDS_5:
                s = seed["data_seed"]
                eid = f"1C_{config['tag']}_n{n_nodes}_seed{s}"
                cfg: dict[str, Any] = {"experiment_id": eid}
                cfg = _deep_merge(cfg, _mode_block())
                cfg["problem"] = {"n_nodes": n_nodes}
                cfg["model"] = dict(config["model"])
                train_block = _ppo_training_block(100_000)
                cfg = _deep_merge(cfg, train_block)
                cfg = _deep_merge(cfg, _critic_block(config["critic"]))
                cfg["quantum"]["enabled"] = False
                cfg = _deep_merge(cfg, {"seeds": seed})
                fname = f"{config['tag']}_n{n_nodes}_seed{s}.yaml"
                _write(phase_dir / fname, cfg, dry_run)
                count += 1

    print(f"  Phase 1C: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Quantum variants
# All use: backend=default.qubit, diff_method=backprop (MANDATORY)
# Strategies: partial encoder, full encoder, encoder+decoder, full quantum, Giang replica
# ─────────────────────────────────────────────────────────────────────────────

QUANTUM_BASE = {
    "quantum": {
        "enabled": True,
        "backend": "default.qubit",
        "diff_method": "backprop",
        "rotation": "RYRZ",
        "topology": "ring",
        "n_qubits_actor": 4,
        "n_qubits_critic": 2,
        "n_layers_quantum_actor": 4,
        "n_layers_quantum_critic": 2,
    }
}

QUANTUM_STRATEGIES = [
    {
        "tag": "lei_q_encoder_fc_node_only",
        "desc": "Lei + quantum fc_node projection only",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": True, "fc_edge": False, "gat_layers": [], "gat_attn_layers": []},
            "actor_decoder": {"fc": False, "fc1": False},
            "critic": {"type": "shared_conv1d", "enabled": False, "layers": []},
        },
    },
    {
        "tag": "lei_q_encoder_gat_layer0",
        "desc": "Lei + quantum GAT layer 0 only",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": False, "fc_edge": False, "gat_layers": [0], "gat_attn_layers": []},
            "actor_decoder": {"fc": False, "fc1": False},
            "critic": {"type": "shared_conv1d", "enabled": False, "layers": []},
        },
    },
    {
        "tag": "lei_q_encoder_all_gat",
        "desc": "Lei + quantum all 4 GAT layers",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": False, "fc_edge": False, "gat_layers": [0, 1, 2, 3], "gat_attn_layers": []},
            "actor_decoder": {"fc": False, "fc1": False},
            "critic": {"type": "shared_conv1d", "enabled": False, "layers": []},
        },
    },
    {
        "tag": "lei_q_encoder_fc_and_gat0",
        "desc": "Lei + quantum fc_node + GAT layer 0",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": True, "fc_edge": False, "gat_layers": [0], "gat_attn_layers": []},
            "actor_decoder": {"fc": False, "fc1": False},
            "critic": {"type": "shared_conv1d", "enabled": False, "layers": []},
        },
    },
    {
        "tag": "lei_q_encoder_actor_decoder",
        "desc": "Lei + quantum encoder (fc_node+GAT0) + actor decoder (fc+fc1)",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": True, "fc_edge": False, "gat_layers": [0], "gat_attn_layers": []},
            "actor_decoder": {"fc": True, "fc1": True, "n_qubits": 4, "n_layers": 4},
            "critic": {"type": "shared_conv1d", "enabled": False, "layers": []},
        },
    },
    {
        "tag": "lei_q_full_quantum_own_critic",
        "desc": "Lei + full quantum encoder + decoder + own_encoder_conv1d critic (quantum)",
        "model_base": "lei",
        "quantum_override": {
            "encoder": {"fc_node": True, "fc_edge": False, "gat_layers": [0, 1, 2, 3], "gat_attn_layers": []},
            "actor_decoder": {"fc": True, "fc1": True, "n_qubits": 4, "n_layers": 4},
            "critic": {
                "type": "own_encoder_conv1d",
                "enabled": True,
                "layers": [],
                "encoder": {"fc_node": True, "fc_edge": False, "gat_layers": [0], "gat_attn_layers": []},
            },
        },
    },
]


def _giang_paper_exact(seed: dict) -> dict:
    """
    Giang et al. 2025 paper-exact configuration.
    3 seeds, n_nodes=20, data_size=100k, backend=default.qubit+backprop.
    """
    s = seed["data_seed"]
    cfg: dict[str, Any] = {
        "experiment_id": f"2_giang_paper_exact_seed{s}",
        "mode": "hybrid_gpu",
        "problem": {"n_nodes": 20},
        "model": {
            "hidden_node_dim": 128,
            "hidden_edge_dim": 16,
            "conv_layers": 4,
            "use_ffn": False,
            "normalization": "batch",
            "tanh_clipping": 10.0,
            "pool_behavior": "static",
        },
        "training": {
            "method": "ppo",
            "policy_mode": "sampling",
            "lr": 3.0e-4,
            "batch_size": 512,
            "epochs": 100,
            "timestep": 1,
            "ppo_epoch": 3,
            "entropy_value": 0.01,
            "eps_clip": 0.2,
        },
        "dataset": {"data_size": 100_000, "val_size": 10_000},
        "quantum": {
            "enabled": True,
            "backend": "default.qubit",
            "diff_method": "backprop",
            "rotation": "RYRZ",
            "topology": "ring",
            "n_qubits_actor": 4,
            "n_qubits_critic": 2,
            "n_layers_quantum_actor": 4,
            "n_layers_quantum_critic": 2,
            "encoder": {
                "fc_node": True,
                "fc_edge": True,
                "gat_layers": [0, 1, 2, 3],
                "gat_attn_layers": [],
            },
            "actor_decoder": {"fc": True, "fc1": True, "n_qubits": 4, "n_layers": 4},
            "critic": {
                "type": "own_encoder_conv1d",
                "enabled": True,
                "layers": [],
                "encoder": {
                    "fc_node": True,
                    "fc_edge": False,
                    "gat_layers": [0],
                    "gat_attn_layers": [],
                },
            },
        },
        "seeds": seed,
    }
    return cfg


def phase2(dry_run: bool) -> None:
    print("\n[Phase 2] Quantum variants …")
    phase_dir = OUTPUT_ROOT / "phase2"
    count = 0

    # ── Quantum strategy ablations ────────────────────────────────────────────
    for strategy in QUANTUM_STRATEGIES:
        for seed in SEEDS_3:
            s = seed["data_seed"]
            eid = f"2_{strategy['tag']}_seed{s}"
            cfg: dict[str, Any] = {"experiment_id": eid}
            cfg = _deep_merge(cfg, _mode_block())
            if strategy["model_base"] == "lei":
                cfg = _deep_merge(cfg, {"problem": {"n_nodes": 20}})
                cfg = _deep_merge(cfg, {"model": _lei_model_block()["model"]})
            cfg = _deep_merge(cfg, _ppo_training_block(100_000))
            # Base quantum block
            cfg = _deep_merge(cfg, QUANTUM_BASE)
            # Strategy-specific quantum override
            for k, v in strategy["quantum_override"].items():
                cfg["quantum"][k] = v
            cfg = _deep_merge(cfg, {"seeds": seed})
            fname = f"{strategy['tag']}_seed{s}.yaml"
            _write(phase_dir / fname, cfg, dry_run)
            count += 1

    # ── Giang paper exact replicas ────────────────────────────────────────────
    for seed in SEEDS_3:
        s = seed["data_seed"]
        cfg = _giang_paper_exact(seed)
        fname = f"giang_paper_exact_seed{s}.yaml"
        _write(phase_dir / fname, cfg, dry_run)
        count += 1

    print(f"  Phase 2: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Optional quantum circuit tuning
# Vary: rotation {RY, RYRZ, RXRYRZ}, topology {ring, brickwall, chain}
# Base: best Phase 2 config (placeholder: lei_q_encoder_fc_and_gat0)
# ─────────────────────────────────────────────────────────────────────────────

def phase3(dry_run: bool) -> None:
    print("\n[Phase 3] Quantum circuit tuning …")
    phase_dir = OUTPUT_ROOT / "phase3"
    count = 0

    rotations = ["RY", "RYRZ", "RXRYRZ"]
    topologies = ["ring", "brickwall", "chain"]

    for rotation in rotations:
        for topology in topologies:
            for seed in SEEDS_3:
                s = seed["data_seed"]
                eid = f"3_rot_{rotation.lower()}_topo_{topology}_seed{s}"
                cfg: dict[str, Any] = {"experiment_id": eid}
                cfg = _deep_merge(cfg, _mode_block())
                cfg = _deep_merge(cfg, {"problem": {"n_nodes": 20}})
                cfg = _deep_merge(cfg, {"model": _lei_model_block()["model"]})
                cfg = _deep_merge(cfg, _ppo_training_block(100_000))
                cfg = _deep_merge(cfg, QUANTUM_BASE)
                # Apply circuit variation
                cfg["quantum"]["rotation"] = rotation
                cfg["quantum"]["topology"] = topology
                # Partial encoder (best Phase 2 config placeholder)
                cfg["quantum"]["encoder"] = {
                    "fc_node": True,
                    "fc_edge": False,
                    "gat_layers": [0],
                    "gat_attn_layers": [],
                }
                cfg["quantum"]["actor_decoder"] = {"fc": False, "fc1": False}
                cfg["quantum"]["critic"] = {
                    "type": "shared_conv1d",
                    "enabled": False,
                    "layers": [],
                }
                cfg = _deep_merge(cfg, {"seeds": seed})
                fname = f"rot_{rotation.lower()}_topo_{topology}_seed{s}.yaml"
                _write(phase_dir / fname, cfg, dry_run)
                count += 1

    print(f"  Phase 3: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# Manifest writer
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(dry_run: bool) -> None:
    if dry_run:
        return
    manifest_path = OUTPUT_ROOT / "manifest.txt"
    lines = ["phase,file,experiment_id"]
    for phase_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not phase_dir.is_dir():
            continue
        phase = phase_dir.name
        for yml in sorted(phase_dir.glob("*.yaml")):
            with open(yml) as f:
                data = yaml.safe_load(f)
            eid = data.get("experiment_id", "?")
            lines.append(f"{phase},{yml.name},{eid}")
    with open(manifest_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nManifest written: {manifest_path.relative_to(BASE_DIR)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PHASE_MAP = {
    "1A": phase1a,
    "1B": phase1b,
    "1C": phase1c,
    "2": phase2,
    "3": phase3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation YAML configs")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=list(PHASE_MAP.keys()),
        default=list(PHASE_MAP.keys()),
        help="Which phases to generate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output paths without writing files",
    )
    args = parser.parse_args()

    print(f"Output root: {OUTPUT_ROOT}")
    for phase in args.phases:
        PHASE_MAP[phase](args.dry_run)

    write_manifest(args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
