"""
Pre-validation of configs before dispatching to SageMaker or running locally.

Usage standalone:
    python validate_config.py configs/base.yaml configs/experiment.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List

import yaml


REQUIRED_KEYS = {
    "problem": ["n_nodes"],
    "dataset": ["data_size", "val_size"],
    "training": ["lr", "batch_size", "epochs"],
}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def validate_configs(
    base_path: Path,
    config_path: Path,
    data_size: int | None = None,
    val_size: int | None = None,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    base_path = Path(str(base_path).replace("\\", "/"))
    config_path = Path(str(config_path).replace("\\", "/"))

    if not base_path.exists():
        errors.append(f"Base config not found: {base_path}")
        return False, errors
    if not config_path.exists():
        errors.append(f"Config not found: {config_path}")
        return False, errors

    try:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in {base_path}: {e}")
        return False, errors

    try:
        with open(config_path) as f:
            exp_cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in {config_path}: {e}")
        return False, errors

    if not isinstance(base_cfg, dict):
        errors.append(f"Base config must be a dict, got {type(base_cfg).__name__}")
        return False, errors
    if not isinstance(exp_cfg, dict):
        errors.append(f"Experiment config must be a dict, got {type(exp_cfg).__name__}")
        return False, errors

    merged = _deep_update(base_cfg, exp_cfg)

    # Required sections/keys
    for section, keys in REQUIRED_KEYS.items():
        if section not in merged:
            errors.append(f"Required section missing: '{section}'")
            continue
        for key in keys:
            if key not in merged[section]:
                errors.append(f"Required key missing: '{section}.{key}'")

    # data_size / val_size
    ds = merged.get("dataset", {})
    effective_data_size = data_size if data_size is not None else ds.get("data_size")
    effective_val_size = val_size if val_size is not None else ds.get("val_size")

    if effective_data_size is not None:
        try:
            if int(effective_data_size) <= 0:
                errors.append(f"data_size must be > 0, got {effective_data_size}")
        except (TypeError, ValueError):
            errors.append(f"data_size is not an integer: {effective_data_size}")

    if effective_val_size is not None:
        try:
            if int(effective_val_size) <= 0:
                errors.append(f"val_size must be > 0, got {effective_val_size}")
        except (TypeError, ValueError):
            errors.append(f"val_size is not an integer: {effective_val_size}")

    # Model dimensions
    model = merged.get("model", {})
    for dim_key in ("hidden_node_dim", "hidden_edge_dim"):
        val = model.get(dim_key)
        if val is not None and (not isinstance(val, int) or val <= 0):
            errors.append(f"model.{dim_key} must be int > 0, got {val}")

    # Quantum params (new config structure: quantum.enabled)
    quantum = merged.get("quantum", {})
    if quantum.get("enabled"):
        for qkey in ("n_qubits_actor", "n_qubits_critic"):
            val = quantum.get(qkey)
            if val is None:
                errors.append(f"quantum.enabled=true but '{qkey}' not defined in quantum section")
            elif not isinstance(val, int) or val <= 0:
                errors.append(f"quantum.{qkey} must be int > 0, got {val}")

    if errors:
        return False, errors
    return True, []


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <base.yaml> <experiment.yaml> [data_size] [val_size]")
        sys.exit(1)

    base_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    data_size = int(sys.argv[3]) if len(sys.argv) > 3 else None
    val_size = int(sys.argv[4]) if len(sys.argv) > 4 else None

    ok, errors = validate_configs(base_path, config_path, data_size, val_size)
    if ok:
        print("OK — configs valid")
    else:
        print("ERROR — invalid configs:")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
