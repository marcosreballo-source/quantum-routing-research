import os
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def configs_dir():
    return CONFIGS_DIR


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def mini_cfg():
    """Minimal config namespace matching base.yaml defaults — ultra-small sizes for tests."""
    return SimpleNamespace(
        mode="hybrid_cpu",
        problem=SimpleNamespace(n_nodes=10),
        dataset=SimpleNamespace(data_size=8, val_size=4),
        model=SimpleNamespace(hidden_node_dim=64, hidden_edge_dim=16, conv_layers=1),
        training=SimpleNamespace(
            method="ppo", policy_mode="sampling",
            lr=3e-4, batch_size=4, epochs=1, timestep=1,
            ppo_epoch=1, entropy_value=0.01, eps_clip=0.2,
            lr_decay=1.0, baseline=None, rollout_baseline_quantum=True,
        ),
        evaluation=SimpleNamespace(
            n_samples=2, eval_size=4,
            temperatures=SimpleNamespace(**{"11": 2.5, "21": 2.5, "51": 1.8, "101": 1.2}),
        ),
        mlflow=SimpleNamespace(tracking_uri=None, experiment_prefix="test"),
        hardware=SimpleNamespace(use_gpu=False, num_workers=0, pin_memory=False),
        seeds=SimpleNamespace(data_seed=42, model_seed=42, random_seed=42, dropout_seed=42, sampling_seed=42),
        dropout=SimpleNamespace(gat=0.0, attn=0.0, attn_output=0.0, decoder=0.0, critic=0.0),
        temperature=SimpleNamespace(initial=2.5, final=2.5, schedule="constant", decay_rate=1.0),
        quantum=SimpleNamespace(
            enabled=False,
            n_qubits_actor=4, n_qubits_critic=4,
            n_layers_quantum_actor=4, n_layers_quantum_critic=2,
            encoder=SimpleNamespace(fc_node=False, fc_edge=False, gat_layers=[]),
            critic=SimpleNamespace(enabled=False, layers=[]),
            decoder=SimpleNamespace(enabled=False),
            backend="lightning.qubit",
        ),
        runtime=SimpleNamespace(
            verbose_level=0, profiler_enabled=False,
            device_preference="cpu", force_classical=False,
            device=torch.device("cpu"),
        ),
    )


@pytest.fixture
def quantum_cfg_full():
    """Quantum config dict as produced by PPO_train.train() / evaluate._build_quantum_cfg()."""
    return {
        'n_qubits_actor': 4,
        'n_layers_quantum_actor': 4,
        'n_qubits_critic': 4,
        'n_layers_quantum_critic': 2,
        'encoder': {
            'fc_node': True,
            'fc_edge': True,
            'gat_layers': [0, 1, 2],
        },
        'critic': {
            'layers': [0, 1, 2],
        },
    }


@pytest.fixture
def dropout_cfg():
    return {
        'gat': 0.0,
        'attn': 0.0,
        'attn_output': 0.0,
        'decoder': 0.0,
        'critic': 0.0,
    }


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Creates a temporary run directory with checkpoints/ and artifacts/ subdirs."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "artifacts").mkdir()
    return run_dir
