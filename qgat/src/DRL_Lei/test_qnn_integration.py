"""
Integration tests for Quantum Neural Network layers in Q-GAT merged codebase.

Tests validate that quantum layers are functioning correctly:
1. Parameter changes during backward pass
2. Quantum vs classical output differences
3. Gradient flow through quantum layers
4. Actor_critic quantum instantiation via quantum_cfg dict
"""

import torch
import logging
import pytest

from DRL_Lei.VRP_PPO_Model import Actor_critic, GatConv, Encoder, Critic
from DRL_Lei.quantum_layers import QuantumLayer, create_quantum_layer

test_logger = logging.getLogger('TestQNN')
test_logger.setLevel(logging.DEBUG)
if not test_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    test_logger.addHandler(handler)


# --- Helpers ---

QUANTUM_CFG_FULL = {
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

DROPOUT_CFG = {
    'gat': 0.0,
    'attn': 0.0,
    'attn_output': 0.0,
    'decoder': 0.0,
    'critic': 0.0,
}


class TestQuantumLayers:
    """Test individual quantum layers."""

    def test_quantum_layer_creation(self):
        layer = QuantumLayer(128, 128, n_qubits=4, n_layers=4)
        x = torch.randn(32, 128)
        output = layer(x)
        assert output.shape == (32, 128)
        assert not torch.isnan(output).any()

    def test_create_quantum_layer_factory(self):
        layer = create_quantum_layer(128, 64, n_qubits=4, n_layers=4)
        x = torch.randn(16, 128)
        output = layer(x)
        assert output.shape == (16, 64)


class TestQuantumParameters:
    """Verify quantum θ parameters change during backward pass."""

    def test_quantum_params_changing_gatconv(self):
        gatconv = GatConv(128, 128, 16, use_quantum=True, n_qubits=4, n_layers_quantum=4)

        initial_params = {
            name: param.clone().detach()
            for name, param in gatconv.named_parameters()
            if 'quantum' in name
        }
        assert len(initial_params) > 0

        x = torch.randn(32, 128, requires_grad=True)
        edge_attr = torch.randn(128, 16)
        edge_index = torch.randint(0, 32, (2, 128))
        output = gatconv(x, edge_index, edge_attr)
        output.sum().backward()

        param_changed = any(
            not torch.allclose(param, initial_params[name], atol=1e-4)
            for name, param in gatconv.named_parameters()
            if 'quantum' in name and param.grad is not None
        )
        assert param_changed, "GatConv quantum params should change during backward"

    def test_quantum_params_changing_critic(self):
        critic = Critic(
            128,
            quantum_critic_cfg=QUANTUM_CFG_FULL['critic'],
            n_qubits_critic=QUANTUM_CFG_FULL['n_qubits_critic'],
            n_layers_quantum_critic=QUANTUM_CFG_FULL['n_layers_quantum_critic'],
            dropout_critic=DROPOUT_CFG['critic'],
        )

        initial_params = {
            name: param.clone().detach()
            for name, param in critic.named_parameters()
            if 'quantum' in name
        }
        assert len(initial_params) > 0

        x = torch.randn(32, 21, 128)
        output = critic(x)
        output.sum().backward()

        param_changed = any(
            not torch.allclose(param, initial_params[name], atol=1e-4)
            for name, param in critic.named_parameters()
            if 'quantum' in name and param.grad is not None
        )
        assert param_changed, "Critic quantum params should change during backward"


class TestQuantumVsClassical:
    """Quantum and classical outputs should differ."""

    def test_quantum_vs_classical_gatconv(self):
        x_test = torch.randn(16, 128)
        edge_attr = torch.randn(64, 16)
        edge_index = torch.randint(0, 128, (2, 64))

        gat_classical = GatConv(128, 128, 16, use_quantum=False)
        out_classical = gat_classical(x_test, edge_index, edge_attr)

        gat_quantum = GatConv(128, 128, 16, use_quantum=True, n_qubits=4, n_layers_quantum=4)
        out_quantum = gat_quantum(x_test, edge_index, edge_attr)

        assert not torch.allclose(out_classical, out_quantum, atol=1e-3)

    def test_quantum_vs_classical_critic(self):
        x_test = torch.randn(8, 21, 128)

        critic_classical = Critic(128, quantum_critic_cfg={}, dropout_critic=DROPOUT_CFG['critic'])
        out_classical = critic_classical(x_test)

        critic_quantum = Critic(
            128,
            quantum_critic_cfg=QUANTUM_CFG_FULL['critic'],
            n_qubits_critic=QUANTUM_CFG_FULL['n_qubits_critic'],
            n_layers_quantum_critic=QUANTUM_CFG_FULL['n_layers_quantum_critic'],
            dropout_critic=DROPOUT_CFG['critic'],
        )
        out_quantum = critic_quantum(x_test)

        assert not torch.allclose(out_classical, out_quantum, atol=1e-2)


class TestGradientFlow:
    """Gradients should flow through quantum layers without NaN/Inf."""

    def test_quantum_gradients_flow_gatconv(self):
        gatconv = GatConv(128, 128, 16, use_quantum=True, n_qubits=4, n_layers_quantum=4)
        x = torch.randn(16, 128, requires_grad=True)
        edge_attr = torch.randn(64, 16)
        edge_index = torch.randint(0, 128, (2, 64))

        output = gatconv(x, edge_index, edge_attr)
        output.sum().backward()

        quantum_grads_found = False
        for name, param in gatconv.named_parameters():
            if 'quantum' in name and param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                quantum_grads_found = True
        assert quantum_grads_found

    def test_quantum_gradients_flow_critic(self):
        critic = Critic(
            128,
            quantum_critic_cfg=QUANTUM_CFG_FULL['critic'],
            n_qubits_critic=QUANTUM_CFG_FULL['n_qubits_critic'],
            n_layers_quantum_critic=QUANTUM_CFG_FULL['n_layers_quantum_critic'],
            dropout_critic=DROPOUT_CFG['critic'],
        )
        x = torch.randn(8, 21, 128, requires_grad=True)
        output = critic(x)
        output.sum().backward()

        quantum_grads_found = False
        for name, param in critic.named_parameters():
            if 'quantum' in name and param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                quantum_grads_found = True
        assert quantum_grads_found


class TestActorCriticIntegration:
    """Actor_critic via quantum_cfg dict."""

    def test_actor_critic_quantum_instantiation(self):
        ac = Actor_critic(
            input_node_dim=3,
            hidden_node_dim=128,
            input_edge_dim=1,
            hidden_edge_dim=16,
            conv_laysers=3,
            quantum_cfg=QUANTUM_CFG_FULL,
            dropout_cfg=DROPOUT_CFG,
        )
        quantum_params = [n for n, _ in ac.named_parameters() if 'quantum' in n]
        assert len(quantum_params) > 0

    def test_actor_critic_classical_instantiation(self):
        ac = Actor_critic(
            input_node_dim=3,
            hidden_node_dim=128,
            input_edge_dim=1,
            hidden_edge_dim=16,
            conv_laysers=3,
            quantum_cfg={},
            dropout_cfg=DROPOUT_CFG,
        )
        quantum_params = [n for n, _ in ac.named_parameters() if 'quantum' in n]
        assert len(quantum_params) == 0
