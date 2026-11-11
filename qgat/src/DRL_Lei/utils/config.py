import sys
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
import multiprocessing


# ==========================================================
# CONFIG LOADING
# ==========================================================

def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Convert nested dict -> attribute object."""
    return SimpleNamespace(**{
        str(k): dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })


def namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Convert nested namespace/dict to plain dict, handling non-serializable objects."""

    if isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    if isinstance(ns, torch.device):
        return str(ns)
    if hasattr(ns, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    # For primitives and other types, return as-is
    return ns


# ==========================================================
# MODE RESOLUTION
# ==========================================================

MODE_PRESETS = {
    "debug": {
        "verbose_level": 2,
        "profiler_enabled": True,
        "device_preference": "auto",
        "force_classical": False,
    },
    "classical": {
        "verbose_level": 0,
        "profiler_enabled": False,
        "device_preference": "auto",
        "force_classical": True,
    },
    "hybrid_cpu": {
        "verbose_level": 1,
        "profiler_enabled": False,
        "device_preference": "cpu",
        "force_classical": False,
    },
    "hybrid_gpu": {
        "verbose_level": 1,
        "profiler_enabled": False,
        "device_preference": "gpu",
        "force_classical": False,
    },
    "production": {
        "verbose_level": 0,
        "profiler_enabled": False,
        "device_preference": "gpu",
        "force_classical": False,
    },
}


def resolve_device(preference: str) -> torch.device:
    if preference == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_mode(cfg: SimpleNamespace) -> SimpleNamespace:
    """Inject runtime.* attributes based on the chosen mode."""
    mode_name = getattr(cfg, "mode", "hybrid_gpu")
    preset = MODE_PRESETS.get(mode_name)
    if preset is None:
        raise ValueError(f"Unknown mode '{mode_name}'. Choose from: {list(MODE_PRESETS)}")

    runtime = SimpleNamespace(
        verbose_level=preset["verbose_level"],
        profiler_enabled=preset["profiler_enabled"],
        device_preference=preset["device_preference"],
        force_classical=preset["force_classical"],
        device=resolve_device(preset["device_preference"]),
    )
    cfg.runtime = runtime

    # If force_classical, override quantum.enabled
    if runtime.force_classical and hasattr(cfg, "quantum"):
        cfg.quantum.enabled = False

    return cfg

def resolve_hardware(cfg: SimpleNamespace) -> SimpleNamespace:
    """Resolve hardware 'auto' values to concrete integers/booleans.
    
    - num_workers: 'auto' → detect CPUs (0 for local dev, 2-4 for GPU, 4-8 for CPU-only)
    - pin_memory: 'auto' → True if CUDA available, False otherwise
    - use_gpu: 'auto' → True if CUDA available, False otherwise
    """

    if not hasattr(cfg, 'hardware'):
        return cfg
    
    hw = cfg.hardware
    has_cuda = torch.cuda.is_available()
    cpu_count = multiprocessing.cpu_count()
    
    # Resolve num_workers
    if hasattr(hw, 'num_workers'):
        if hw.num_workers == "auto":
            if has_cuda:
                # GPU + torch_geometric: MUST use num_workers=0
                # (spawning subprocesses + CUDA = initialization errors in SageMaker)
                hw.num_workers = 0
            else:
                # CPU only: use workers for I/O parallelism
                hw.num_workers = max(4, min(cpu_count - 1, 8))
        elif isinstance(hw.num_workers, str):
            try:
                hw.num_workers = int(hw.num_workers)
            except ValueError:
                hw.num_workers = 0
    
    # Resolve pin_memory
    if hasattr(hw, 'pin_memory'):
        if hw.pin_memory == "auto":
            # Pin memory only for CPU-only (cannot pin CUDA tensors)
            hw.pin_memory = not has_cuda
        elif isinstance(hw.pin_memory, str):
            hw.pin_memory = hw.pin_memory.lower() in ("true", "1", "yes")
    
    # Resolve use_gpu
    if hasattr(hw, 'use_gpu'):
        if hw.use_gpu == "auto":
            hw.use_gpu = has_cuda
        elif isinstance(hw.use_gpu, str):
            hw.use_gpu = hw.use_gpu.lower() in ("true", "1", "yes")
    
    return cfg


# ==========================================================
# QUANTUM CONFIG DEFAULTS
# ==========================================================

_QUANTUM_DEFAULTS = {
    "enabled": False,
    "rotation": "RYRZ",
    "topology": "ring",
    "n_qubits_actor": 4,
    "n_qubits_critic": 4,
    "n_layers_quantum_actor": 4,
    "n_layers_quantum_critic": 2,
    "encoder": {"fc_node": False, "fc_edge": False, "gat_layers": []},
    "critic": {"enabled": False, "layers": [], "type": "shared_conv1d",
               "encoder": {"fc_node": False, "fc_edge": False, "gat_layers": []}},
    "actor_decoder": {"fc": False, "fc1": False, "n_qubits": 4, "n_layers": 4},
    "backend": "lightning.qubit",
}

_SEEDS_DEFAULTS = {
    "data_seed": 42,
    "model_seed": 42,
    "random_seed": 42,
    "dropout_seed": 42,
    "sampling_seed": 42,
}

_DROPOUT_DEFAULTS = {
    "gat": 0.0,
    "attn": 0.0,
    "attn_output": 0.0,
    "decoder": 0.0,
    "critic": 0.0,
}

_TEMPERATURE_DEFAULTS = {
    "initial": 2.5,
    "final": 2.5,
    "schedule": "constant",
    "decay_rate": 1.0,
}

_HARDWARE_DEFAULTS = {
    "use_gpu": "auto",
    "num_workers": "auto",
    "pin_memory": "auto",
}


def _ensure_section(cfg_dict: dict, section: str, defaults: dict):
    if section not in cfg_dict:
        cfg_dict[section] = {}
    for k, v in defaults.items():
        if k not in cfg_dict[section]:
            if isinstance(v, dict):
                cfg_dict[section][k] = dict(v)
            else:
                cfg_dict[section][k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if kk not in cfg_dict[section][k]:
                    cfg_dict[section][k][kk] = vv


def apply_defaults(cfg_dict: dict) -> dict:
    _ensure_section(cfg_dict, "quantum", _QUANTUM_DEFAULTS)
    _ensure_section(cfg_dict, "seeds", _SEEDS_DEFAULTS)
    _ensure_section(cfg_dict, "dropout", _DROPOUT_DEFAULTS)
    _ensure_section(cfg_dict, "temperature", _TEMPERATURE_DEFAULTS)
    _ensure_section(cfg_dict, "hardware", _HARDWARE_DEFAULTS)

    cfg_dict.setdefault("mode", "hybrid_gpu")

    training = cfg_dict.setdefault("training", {})
    training.setdefault("method", "ppo")
    training.setdefault("baseline", None)
    training.setdefault("bl_warmup_epochs", 1)
    training.setdefault("bl_alpha", 0.05)
    training.setdefault("exp_beta", 0.8)
    training.setdefault("lr_critic", None)
    training.setdefault("rollout_baseline_quantum", True)
    training.setdefault("lr_decay", 1.0)
    training.setdefault("loss_weights", {"policy": 1.0, "value": 0.5, "entropy": 0.01})

    dataset = cfg_dict.setdefault("dataset", {})
    dataset.setdefault("distribution", "uniform")

    model = cfg_dict.setdefault("model", {})
    model.setdefault("use_ffn", False)
    model.setdefault("normalization", "batch")
    model.setdefault("tanh_clipping", 10.0)
    model.setdefault("pool_behavior", "static")

    evaluation = cfg_dict.setdefault("evaluation", {})
    evaluation.setdefault("enabled", True)
    evaluation.setdefault("modes", ["greedy", "sampling"])
    evaluation.setdefault("n_samples", 1280)
    evaluation.setdefault("eval_size", 1000)
    evaluation.setdefault("beam_width", 0)
    evaluation.setdefault("temperatures", {21: 2.5, 51: 1.8, 101: 1.2})

    cfg_dict.setdefault("mlflow", {"tracking_uri": None, "experiment_prefix": "DRL_Lei", "backend_store": None, "artifact_root": None})
    return cfg_dict


# ==========================================================
# CONFIG LOADERS
# ==========================================================

def load_config(base_path: Path, exp_path: Path) -> SimpleNamespace:
    with open(base_path, encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    with open(exp_path, encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    merged = deep_update(base_cfg, exp_cfg)
    merged = apply_defaults(merged)
    cfg = dict_to_namespace(merged)
    cfg = resolve_mode(cfg)
    cfg = resolve_hardware(cfg)
    return cfg


def load_single_config(path: Path) -> SimpleNamespace:
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    raw = apply_defaults(raw)
    cfg = dict_to_namespace(raw)
    cfg = resolve_mode(cfg)
    cfg = resolve_hardware(cfg)
    return cfg


def load_eval_config(base_cfg: SimpleNamespace, eval_path: Path) -> SimpleNamespace:
    """Override only the evaluation block from an optional eval config file."""
    with open(eval_path, encoding='utf-8') as f:
        eval_cfg = yaml.safe_load(f) or {}
    merged = deep_update(namespace_to_dict(base_cfg), eval_cfg)
    merged = apply_defaults(merged)
    cfg = dict_to_namespace(merged)
    cfg = resolve_mode(cfg)
    cfg = resolve_hardware(cfg)
    return cfg


def validate_params(params: dict, expected_params: list):
    for param in expected_params:
        if param not in params:
            print(f"Missing required parameter: {param}")
            sys.exit(1)
