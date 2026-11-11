import pytest
from pathlib import Path

from validate_config import validate_configs, _deep_update, REQUIRED_KEYS


class TestDeepUpdate:
    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        result = _deep_update(base, {"a": {"c": 99}})
        assert result == {"a": {"b": 1, "c": 99}}

    def test_override_non_dict(self):
        base = {"a": 1}
        result = _deep_update(base, {"a": 2})
        assert result == {"a": 2}


class TestValidateConfigs:
    def test_valid_config(self, configs_dir):
        ok, errors = validate_configs(
            configs_dir / "base.yaml",
            configs_dir / "light_test.yaml",
        )
        assert ok is True
        assert errors == []

    def test_missing_base(self, tmp_path):
        ok, errors = validate_configs(
            tmp_path / "nonexistent.yaml",
            tmp_path / "also_nonexistent.yaml",
        )
        assert ok is False
        assert any("not found" in e for e in errors)

    def test_missing_required_keys(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("mode: debug\n")
        exp = tmp_path / "exp.yaml"
        exp.write_text("mode: debug\n")
        ok, errors = validate_configs(base, exp)
        assert ok is False
        assert any("Required" in e for e in errors)

    def test_invalid_data_size(self, tmp_path, configs_dir):
        ok, errors = validate_configs(
            configs_dir / "base.yaml",
            configs_dir / "light_test.yaml",
            data_size=-1,
        )
        assert ok is False
        assert any("data_size" in e for e in errors)

    def test_quantum_without_qubits(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(
            "problem:\n  n_nodes: 10\n"
            "dataset:\n  data_size: 10\n  val_size: 5\n"
            "training:\n  lr: 0.001\n  batch_size: 4\n  epochs: 1\n"
            "quantum:\n  enabled: true\n"
        )
        exp = tmp_path / "exp.yaml"
        exp.write_text("{}\n")
        ok, errors = validate_configs(base, exp)
        assert ok is False
        assert any("n_qubits" in e for e in errors)

    def test_data_size_override(self, configs_dir):
        ok, errors = validate_configs(
            configs_dir / "base.yaml",
            configs_dir / "light_test.yaml",
            data_size=500,
            val_size=100,
        )
        assert ok is True
