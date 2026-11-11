"""Hook de nome de job do QGAT para o smlaunch (job.name_hook: smlaunch_naming:job_name).

Codifica no nome o que distingue os experimentos de ablação:
    {project}-{squad}-{method}-{baseline}-[{critic}]-{C | Q<topologia>-<ansatz>}
Ex.: bqc-qgat-ppo-critic-sc-C, bqc-qgat-rollout-rollout-Qbw-RXRYRZ
"""

from __future__ import annotations

from pathlib import Path

import yaml

CRITIC_ABBREV = {
    "shared_conv1d": "sc", "shared_fc": "sf",
    "own_encoder_conv1d": "oec", "own_encoder_mlp": "oem",
    "lei_conv1d": "sc", "kool_encoder": "oem",  # nomes legados
}
TOPO_ABBREV = {"brickwall": "bw", "chain": "chn", "lambda": "lmb"}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def job_name(variables: dict) -> str:
    cfg = _merge(_load(variables["base"]), _load(variables["config"]))
    sm = cfg.get("sagemaker") or {}
    training = cfg.get("training") or {}
    quantum = cfg.get("quantum") or {}

    method = str(training.get("method") or "ppo").lower()
    baseline = training.get("baseline") or {"ppo": "critic", "rollout": "rollout"}.get(method, "none")
    baseline = str(baseline).lower()

    parts = [variables.get("project") or sm.get("project", "proj"),
             variables.get("squad") or sm.get("squad", "squad"), method, baseline]

    critic = quantum.get("critic") or {}
    if baseline == "critic" or critic.get("enabled", False):
        raw = critic.get("type") or "shared_conv1d"
        parts.append(CRITIC_ABBREV.get(raw, raw[:4]))

    if quantum.get("enabled", False):
        topo = str(quantum.get("topology") or "ring").lower()
        parts.append(f"Q{TOPO_ABBREV.get(topo, topo)}-{str(quantum.get('rotation') or 'RYRZ').upper()}")
    else:
        parts.append("C")
    return "-".join(parts)
