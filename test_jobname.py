#!/usr/bin/env python
"""Quick test for job name generation."""
import yaml
from launch_sagemaker import _extract_specs_from_config, _build_job_name, _deep_merge

# Load configs
with open('configs/base.yaml') as f:
    base = yaml.safe_load(f) or {}
with open('configs/phase0/giang_original.yaml') as f:
    exp = yaml.safe_load(f) or {}

# Deep merge (como faz launch_sagemaker.py)
def deep_merge(base_dict, override_dict):
    result = base_dict.copy()
    for key, val in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result

cfg = deep_merge(base, exp)

# Test
specs = _extract_specs_from_config(cfg)
print("=" * 60)
print("Specs extraídos do config mesclado:")
print("=" * 60)
for k, v in specs.items():
    print(f"  {k:20} : {v}")

job_name = _build_job_name(cfg, 'bqc', 'qgat')
print("\n" + "=" * 60)
print(f"Nome do job gerado: {job_name}")
print("=" * 60)

# Verify
expected = "bqc-qgat-ppo-critic-oec-Qring-RYRZ"
if job_name == expected:
    print(f"✅ CORRETO! Nome corresponde ao esperado.")
else:
    print(f"❌ ERRO! Esperado: {expected}")
