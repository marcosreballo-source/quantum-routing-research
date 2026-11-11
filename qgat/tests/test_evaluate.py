import pytest
from types import SimpleNamespace

from DRL_Lei.evaluate import _build_quantum_cfg, _build_dropout_cfg


class TestBuildQuantumCfg:
    def test_disabled_returns_empty(self):
        cfg = SimpleNamespace(quantum=SimpleNamespace(enabled=False))
        assert _build_quantum_cfg(cfg) == {}

    def test_missing_quantum_returns_empty(self):
        cfg = SimpleNamespace()
        assert _build_quantum_cfg(cfg) == {}

    def test_enabled_extracts_all_fields(self):
        cfg = SimpleNamespace(quantum=SimpleNamespace(
            enabled=True,
            n_qubits_actor=8,
            n_layers_quantum_actor=3,
            n_qubits_critic=6,
            n_layers_quantum_critic=1,
            encoder=SimpleNamespace(fc_node=True, fc_edge=False, gat_layers=[0, 1]),
            critic=SimpleNamespace(layers=[0]),
        ))
        result = _build_quantum_cfg(cfg)
        assert result['n_qubits_actor'] == 8
        assert result['encoder']['fc_node'] is True
        assert result['encoder']['gat_layers'] == [0, 1]
        assert result['critic']['layers'] == [0]

    def test_enabled_defaults(self):
        cfg = SimpleNamespace(quantum=SimpleNamespace(
            enabled=True,
            encoder=None,
            critic=None,
        ))
        result = _build_quantum_cfg(cfg)
        assert result['encoder']['fc_node'] is False
        assert result['encoder']['gat_layers'] == []
        assert result['critic']['layers'] == []


class TestBuildDropoutCfg:
    def test_returns_all_keys(self):
        cfg = SimpleNamespace(dropout=SimpleNamespace(
            gat=0.1, attn=0.2, attn_output=0.3, decoder=0.4, critic=0.5,
        ))
        result = _build_dropout_cfg(cfg)
        assert result == {'gat': 0.1, 'attn': 0.2, 'attn_output': 0.3, 'decoder': 0.4, 'critic': 0.5}

    def test_missing_dropout_returns_empty(self):
        cfg = SimpleNamespace()
        assert _build_dropout_cfg(cfg) == {}
