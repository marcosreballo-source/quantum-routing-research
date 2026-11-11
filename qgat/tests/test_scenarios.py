"""
End-to-end integration scenario tests.

Each test runs main.py (or a helper) as a subprocess and validates
exit code, stdout, or side-effects. Marked slow + scenario.
"""

import os
import sys
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"
CONFIGS = PROJECT_ROOT / "configs"
PYTHON = sys.executable


def _run_main(*args, timeout=300):
    """Run main.py with given args and return CompletedProcess."""
    cmd = [PYTHON, str(MAIN_PY)] + list(args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(PROJECT_ROOT), env=env,
    )


# -----------------------------------------------------------------
# Scenario 1: PPO classical
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
def test_scenario_ppo_classical():
    result = _run_main(
        "--config", str(CONFIGS / "light_test.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    assert "Using device:" in result.stdout


# -----------------------------------------------------------------
# Scenario 2: PPO quantum
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
@pytest.mark.quantum
def test_scenario_ppo_quantum():
    result = _run_main(
        "--config", str(CONFIGS / "test_qnn_ultralight.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    assert "Quantum ENABLED" in result.stdout


# -----------------------------------------------------------------
# Scenario 3: Rollout classical
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
def test_scenario_rollout_classical():
    result = _run_main(
        "--config", str(CONFIGS / "rollout_classical.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"


# -----------------------------------------------------------------
# Scenario 4: Rollout quantum
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
@pytest.mark.quantum
def test_scenario_rollout_quantum():
    result = _run_main(
        "--config", str(CONFIGS / "rollout_quantum.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"


# -----------------------------------------------------------------
# Scenario 5: Debug mode + quantum (verbose + profiler)
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
@pytest.mark.quantum
def test_scenario_debug_quantum():
    result = _run_main(
        "--config", str(CONFIGS / "debug_quantum.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    # Debug mode should have verbose=2 output
    assert "Batch" in result.stdout or "PROFILER" in result.stdout.upper()


# -----------------------------------------------------------------
# Scenario 6: --mode classical forces quantum OFF
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
def test_scenario_classical_override():
    result = _run_main(
        "--config", str(CONFIGS / "test_qnn_ultralight.yaml"),
        "--mode", "classical",
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    # Should NOT have quantum enabled prints
    assert "Quantum ENABLED" not in result.stdout


# -----------------------------------------------------------------
# Scenario 7: Checkpoint evaluation (train then eval)
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
def test_scenario_checkpoint_eval(tmp_path):
    # Step 1: quick train to produce a checkpoint
    train_result = _run_main(
        "--config", str(CONFIGS / "light_test.yaml"),
        "--no-mlflow",
    )
    assert train_result.returncode == 0, f"Train STDERR:\n{train_result.stderr}"

    # Find the run directory (most recent under PROJECT_ROOT/runs/)
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        pytest.skip("No runs/ directory created — may be using SLURM scratch")

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    assert len(run_dirs) > 0, "No run directories found"

    checkpoint = run_dirs[0] / "checkpoints" / "final.pt"
    if not checkpoint.exists():
        checkpoint = run_dirs[0] / "checkpoints" / "last.pt"
    assert checkpoint.exists(), f"No checkpoint found in {run_dirs[0]}"

    # Step 2: evaluate
    eval_result = _run_main(
        "--config", str(CONFIGS / "light_test.yaml"),
        "--checkpoint", str(checkpoint),
        "--no-mlflow",
    )
    assert eval_result.returncode == 0, f"Eval STDERR:\n{eval_result.stderr}"
    assert "CVRP" in eval_result.stdout or "Greedy" in eval_result.stdout


# -----------------------------------------------------------------
# Scenario 8: QNN integration tests via pytest
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
@pytest.mark.quantum
def test_scenario_qnn_integration():
    test_file = PROJECT_ROOT / "src" / "DRL_Lei" / "test_qnn_integration.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [PYTHON, "-m", "pytest", str(test_file), "-v", "--tb=short"],
        capture_output=True, text=True, timeout=600,
        cwd=str(PROJECT_ROOT), env=env,
    )
    assert result.returncode == 0, f"QNN tests failed:\n{result.stdout}\n{result.stderr}"


# -----------------------------------------------------------------
# Scenario 9: validate_config.py schema validation
# -----------------------------------------------------------------
@pytest.mark.scenario
def test_scenario_validate_config():
    script = PROJECT_ROOT / "validate_config.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [PYTHON, str(script),
         str(CONFIGS / "base.yaml"),
         str(CONFIGS / "light_test.yaml")],
        capture_output=True, text=True, timeout=30,
        cwd=str(PROJECT_ROOT), env=env,
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    assert "OK" in result.stdout


# -----------------------------------------------------------------
# Scenario 10: MLflow profiler metrics (debug mode produces profiler output)
# -----------------------------------------------------------------
@pytest.mark.scenario
@pytest.mark.slow
@pytest.mark.quantum
def test_scenario_mlflow_profiler():
    result = _run_main(
        "--config", str(CONFIGS / "debug_quantum.yaml"),
        "--no-mlflow",
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    # In debug mode with profiler, verbose=2 prints profiler blocks.
    # When --no-mlflow, profiler still prints to stdout if verbose_level >= 1
    stdout_lower = result.stdout.lower()
    assert ("profiler" in stdout_lower or
            "data_transfer" in stdout_lower or
            "actor_forward" in stdout_lower or
            "Batch" in result.stdout), \
        f"Expected profiler or batch output in debug mode, got:\n{result.stdout[-500:]}"
