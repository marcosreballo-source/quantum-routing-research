import pytest
import torch

from DRL_Lei.quantum_layers import QuantumLayer, create_quantum_layer


@pytest.mark.quantum
class TestQuantumLayer:
    def test_forward_shape(self):
        layer = QuantumLayer(32, 32, n_qubits=4, n_layers=2)
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_no_nan(self):
        layer = QuantumLayer(16, 16, n_qubits=4, n_layers=1)
        x = torch.randn(4, 16)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_different_in_out(self):
        layer = QuantumLayer(32, 16, n_qubits=4, n_layers=2)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_has_parameters(self):
        layer = QuantumLayer(32, 32, n_qubits=4, n_layers=2)
        params = list(layer.parameters())
        assert len(params) > 0


@pytest.mark.quantum
class TestCreateQuantumLayer:
    def test_default_returns_module(self):
        layer = create_quantum_layer(32, 16, n_qubits=4, n_layers=2)
        assert isinstance(layer, torch.nn.Module)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        layer = create_quantum_layer(32, 32, n_qubits=4, n_layers=2)
        x = torch.randn(4, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None

    def test_param_count_positive(self):
        layer = create_quantum_layer(32, 32, n_qubits=4, n_layers=2)
        total = sum(p.numel() for p in layer.parameters())
        assert total > 0
