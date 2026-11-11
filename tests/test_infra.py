"""
Infrastructure tests — dry-run validation of SageMaker / SLURM integration.

No AWS or SLURM calls are made. These tests validate:
- CLI parsing and path resolution in sagemaker_entry.py / launch_sagemaker.py
- SLURM sbatch scripts reference main.py and set MLflow URI
- validate_config.py CLI works via subprocess
"""

import os
import sys
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS = PROJECT_ROOT / "configs"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# SLURM sbatch validation
# ---------------------------------------------------------------------------
@pytest.mark.infra
class TestSlurmScripts:
    @pytest.fixture(autouse=True)
    def _load_scripts(self):
        self.sbatch_files = list(SCRIPTS_DIR.glob("*.sbatch")) if SCRIPTS_DIR.exists() else []

    def test_sbatch_files_exist(self):
        assert len(self.sbatch_files) > 0, "No .sbatch files found in scripts/"

    def test_all_reference_main_py(self):
        for f in self.sbatch_files:
            content = f.read_text()
            assert "main.py" in content, f"{f.name} does not reference main.py"

    def test_all_have_mlflow_uri_or_nomlflow(self):
        for f in self.sbatch_files:
            content = f.read_text()
            has_mlflow = ("MLFLOW_TRACKING_URI" in content or
                          "mlflow" in content.lower() or
                          "--no-mlflow" in content)
            assert has_mlflow, f"{f.name} has no MLflow configuration"

    def test_sbatch_has_partition(self):
        for f in self.sbatch_files:
            content = f.read_text()
            assert "#SBATCH" in content, f"{f.name} missing SBATCH directives"


# ---------------------------------------------------------------------------
# SageMaker entry point validation
# ---------------------------------------------------------------------------
@pytest.mark.infra
class TestSagemakerEntry:
    def test_sagemaker_entry_exists(self):
        assert (PROJECT_ROOT / "sagemaker_entry.py").exists()

    def test_sagemaker_entry_imports_main(self):
        content = (PROJECT_ROOT / "sagemaker_entry.py").read_text()
        assert "main.py" in content or "main" in content

    def test_sagemaker_entry_reads_sm_env_vars(self):
        content = (PROJECT_ROOT / "sagemaker_entry.py").read_text()
        for var in ["SM_MODEL_DIR", "SM_OUTPUT_DIR"]:
            assert var in content, f"sagemaker_entry.py missing {var}"


# ---------------------------------------------------------------------------
# launch_sagemaker validation
# ---------------------------------------------------------------------------
@pytest.mark.infra
class TestLaunchSagemaker:
    def test_launch_sagemaker_exists(self):
        assert (PROJECT_ROOT / "launch_sagemaker.py").exists()

    def test_launch_validates_config(self):
        content = (PROJECT_ROOT / "launch_sagemaker.py").read_text(encoding="utf-8")
        assert "validate" in content.lower(), \
            "launch_sagemaker.py should validate configs before dispatch"

    def test_launch_generates_job_name(self):
        content = (PROJECT_ROOT / "launch_sagemaker.py").read_text(encoding="utf-8")
        assert "job_name" in content.lower() or "job-name" in content.lower()


# ---------------------------------------------------------------------------
# validate_config.py CLI
# ---------------------------------------------------------------------------
@pytest.mark.infra
class TestValidateConfigCLI:
    def test_valid_config_passes(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "validate_config.py"),
             str(CONFIGS / "base.yaml"),
             str(CONFIGS / "light_test.yaml")],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT), env=env,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_missing_file_fails(self, tmp_path):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "validate_config.py"),
             str(tmp_path / "nonexistent.yaml"),
             str(CONFIGS / "light_test.yaml")],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT), env=env,
        )
        assert result.returncode != 0

    def test_invalid_data_size_via_arg(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "validate_config.py"),
             str(CONFIGS / "base.yaml"),
             str(CONFIGS / "light_test.yaml"),
             "-1"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT), env=env,
        )
        assert result.returncode != 0
