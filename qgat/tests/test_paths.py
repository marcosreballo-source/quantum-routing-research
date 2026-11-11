import pytest
import os
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from DRL_Lei.utils.paths import make_run_dir, make_eval_dir, get_path_suffix


class TestGetPathSuffix:
    def test_contains_date(self):
        suffix = get_path_suffix()
        assert "-" in suffix  # date format YYYY-MM-DD
        parts = suffix.split("_")
        assert len(parts) >= 2  # date + time

    def test_includes_slurm_job_id(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            suffix = get_path_suffix()
            assert "12345" in suffix

    def test_no_slurm(self):
        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        with patch.dict(os.environ, env, clear=True):
            suffix = get_path_suffix()
            assert suffix.count("_") >= 1  # just date_time


class TestMakeRunDir:
    def test_creates_subdirs(self, tmp_path):
        cfg = SimpleNamespace(training=SimpleNamespace(method="ppo"))
        env = os.environ.copy()
        env.pop("TMP_DIR", None)
        env.pop("SLURM_JOB_NAME", None)
        env.pop("SLURM_JOB_ID", None)
        with patch("DRL_Lei.utils.paths._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            run_dir = make_run_dir(cfg)
            assert (run_dir / "checkpoints").is_dir()
            assert (run_dir / "artifacts").is_dir()
            assert "ppo" in run_dir.name

    def test_uses_tmp_dir(self, tmp_path):
        cfg = SimpleNamespace(training=SimpleNamespace(method="rollout"))
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        with patch.dict(os.environ, {"TMP_DIR": str(scratch)}):
            run_dir = make_run_dir(cfg)
            assert str(scratch) in str(run_dir)


class TestMakeEvalDir:
    def test_creates_eval_subdir(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "final.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        eval_dir = make_eval_dir(ckpt)
        assert eval_dir.is_dir()
        assert eval_dir.name == "eval"
