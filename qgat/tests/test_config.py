import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from DRL_Lei.utils.config import (
    deep_update, dict_to_namespace, namespace_to_dict,
    resolve_device, resolve_mode, apply_defaults,
    load_config, load_single_config, validate_params,
    MODE_PRESETS,
)


class TestDeepUpdate:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        result = deep_update(base, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        result = deep_update(base, {"x": {"b": 99}})
        assert result["x"] == {"a": 1, "b": 99}

    def test_add_new_key(self):
        base = {"a": 1}
        result = deep_update(base, {"z": 42})
        assert result == {"a": 1, "z": 42}

    def test_override_dict_with_scalar(self):
        base = {"x": {"a": 1}}
        result = deep_update(base, {"x": 5})
        assert result["x"] == 5

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_update(base, {})
        assert result == {"a": 1}


class TestDictToNamespace:
    def test_flat(self):
        ns = dict_to_namespace({"a": 1, "b": "hello"})
        assert ns.a == 1
        assert ns.b == "hello"

    def test_nested(self):
        ns = dict_to_namespace({"outer": {"inner": 42}})
        assert ns.outer.inner == 42

    def test_roundtrip(self):
        d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        ns = dict_to_namespace(d)
        result = namespace_to_dict(ns)
        assert result == d


class TestResolveDevice:
    def test_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto(self):
        d = resolve_device("auto")
        assert d in (torch.device("cpu"), torch.device("cuda"))

    def test_gpu_fallback(self):
        d = resolve_device("gpu")
        if not torch.cuda.is_available():
            assert d == torch.device("cpu")


class TestResolveMode:
    @pytest.mark.parametrize("mode_name", list(MODE_PRESETS.keys()))
    def test_all_modes_resolve(self, mode_name):
        cfg = SimpleNamespace(mode=mode_name, quantum=SimpleNamespace(enabled=True))
        result = resolve_mode(cfg)
        assert hasattr(result, "runtime")
        assert hasattr(result.runtime, "device")

    def test_classical_forces_quantum_off(self):
        cfg = SimpleNamespace(mode="classical", quantum=SimpleNamespace(enabled=True))
        resolve_mode(cfg)
        assert cfg.quantum.enabled is False

    def test_debug_enables_profiler(self):
        cfg = SimpleNamespace(mode="debug")
        resolve_mode(cfg)
        assert cfg.runtime.profiler_enabled is True
        assert cfg.runtime.verbose_level == 2

    def test_unknown_mode_raises(self):
        cfg = SimpleNamespace(mode="invalid_mode_xyz")
        with pytest.raises(ValueError, match="Unknown mode"):
            resolve_mode(cfg)


class TestApplyDefaults:
    def test_fills_missing_sections(self):
        result = apply_defaults({})
        assert "quantum" in result
        assert "seeds" in result
        assert "dropout" in result
        assert "temperature" in result
        assert "hardware" in result
        assert result["quantum"]["enabled"] is False

    def test_preserves_existing(self):
        cfg = {"quantum": {"enabled": True, "n_qubits_actor": 8}}
        result = apply_defaults(cfg)
        assert result["quantum"]["enabled"] is True
        assert result["quantum"]["n_qubits_actor"] == 8
        assert result["quantum"]["n_qubits_critic"] == 4  # filled default

    def test_training_defaults(self):
        result = apply_defaults({})
        assert result["training"]["method"] == "ppo"
        assert result["training"]["baseline"] is None


class TestLoadConfig:
    def test_load_base_with_light(self, configs_dir):
        cfg = load_config(configs_dir / "base.yaml", configs_dir / "light_test.yaml")
        assert cfg.dataset.data_size == 100
        assert cfg.dataset.val_size == 100
        assert cfg.training.epochs == 2
        assert hasattr(cfg, "runtime")

    def test_load_single(self, configs_dir):
        cfg = load_single_config(configs_dir / "test_qnn_ultralight.yaml")
        assert cfg.quantum.enabled is True
        assert cfg.problem.n_nodes == 10
        assert hasattr(cfg, "runtime")


class TestValidateParams:
    def test_passes_with_all_params(self):
        params = {"lr": 1e-3, "batch_size": 32}
        validate_params(params, ["lr", "batch_size"])

    def test_exits_on_missing(self):
        with pytest.raises(SystemExit):
            validate_params({"lr": 1e-3}, ["lr", "batch_size"])
