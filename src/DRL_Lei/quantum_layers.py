"""
Quantum Layer Implementation for QGAT
Provides QuantumLayer for Actor and Critic networks (PennyLane required)
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from DRL_Lei.ansatz import ansatz as pce_ansatz, parse_rot_sequence

# Setup logger
quantum_logger = logging.getLogger('QuantumLayers')
quantum_logger.setLevel(logging.INFO)

# Check PennyLane availability (backend selection is now per-layer via config)
PL_AVAILABLE = False
try:
    import pennylane as qml
    PL_AVAILABLE = True
    quantum_logger.info("PennyLane available — backend/diff_method configured via YAML")
except ImportError:
    quantum_logger.warning("PennyLane not installed — quantum layers require PennyLane")

# Valid backends and diff_methods
VALID_BACKENDS = {'lightning.gpu', 'lightning.qubit', 'lightning.kokkos', 'default.qubit'}
VALID_DIFF_METHODS = {'best', 'adjoint', 'backprop', 'parameter-shift', 'finite-diff'}

# Runtime info populated on first forward pass (for MLflow tagging)
_RUNTIME_QUANTUM_INFO = {}


def _validate_backend_diff_method(backend: str, diff_method: str) -> tuple[str, str]:
    """Validate and auto-correct incompatible backend/diff_method combinations.

    Returns (backend, diff_method) — possibly corrected with warnings.
    """
    if backend not in VALID_BACKENDS:
        quantum_logger.warning(f"Unknown backend '{backend}', falling back to 'lightning.qubit'")
        backend = 'lightning.qubit'

    if diff_method not in VALID_DIFF_METHODS:
        quantum_logger.warning(f"Unknown diff_method '{diff_method}', falling back to 'best'")
        diff_method = 'best'

    # backprop only works with default.qubit (statevector sim)
    if diff_method == 'backprop' and backend != 'default.qubit':
        quantum_logger.warning(
            f"diff_method='backprop' is incompatible with backend='{backend}'. "
            f"Switching to diff_method='adjoint'. Use backend='default.qubit' for backprop."
        )
        diff_method = 'adjoint'

    # adjoint does not work with default.qubit in torch interface reliably
    # (it works but is slower than backprop; warn user)
    if diff_method == 'adjoint' and backend == 'default.qubit':
        quantum_logger.info(
            "diff_method='adjoint' with backend='default.qubit': consider 'backprop' for better performance."
        )

    return backend, diff_method


def get_runtime_quantum_info() -> dict:
    """Return runtime quantum backend/diff_method info (populated after first forward pass)."""
    return dict(_RUNTIME_QUANTUM_INFO)


class QuantumLayer(nn.Module):
    """
    Full PennyLane quantum layer implementation.
    
    Architecture:
    - Encoding: RY(π*x/2) for each input feature onto qubits
    - VQC: PCE ansatz with configurable rotation/topology
    - Measurement: ⟨Z⟩ expectation values
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (mapped to qubits)
        n_qubits: Number of qubits (default: 4)
        n_layers: Number of VQC layers (default: 4)
        use_bias: Whether to use classical bias (default: True)
        rotation: Rotation gate sequence, e.g. 'RYRZ', 'RXRYRZ' (default: 'RYRZ')
        topology: Entanglement topology — 'ring', 'brickwall', 'chain', 'lambda' (default: 'ring')
        backend: PennyLane device — 'lightning.gpu', 'lightning.qubit', etc. (default: 'lightning.qubit')
        diff_method: Gradient method — 'best', 'adjoint', 'backprop', etc. (default: 'best')
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_qubits: int = 4,
        n_layers: int = 4,
        use_bias: bool = True,
        rotation: str = 'RYRZ',
        topology: str = 'ring',
        backend: str = 'lightning.qubit',
        diff_method: str = 'best',
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.rotation = rotation
        self.topology = topology

        # Validate and store backend/diff_method from config
        self.backend, self.diff_method = _validate_backend_diff_method(backend, diff_method)
        self._logged_runtime_info = False  # log actual device on first forward pass
        
        rot_seq = parse_rot_sequence(rotation)
        self.n_rot_gates = len(rot_seq)
        
        # Encoding layer: project inputs to qubits
        self.fc_in = nn.Linear(in_features, n_qubits)
        
        # Quantum parameters: θ shape (n_layers, n_qubits, n_rot_gates)
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, self.n_rot_gates) * 0.1
        )
        
        # Output projection: qubits back to out_features
        self.fc_out = nn.Linear(n_qubits, out_features)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.orthogonal_(self.theta, gain=0.1)
        if use_bias:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer.
        
        Args:
            x: Input tensor, shape (..., in_features)
            
        Returns:
            Output tensor, shape (..., out_features)
        """
        batch_shape = x.shape[:-1]
        
        # Encode: project to qubit space
        x_encoded = self.fc_in(x)  # (..., n_qubits)
        x_encoded = torch.tanh(x_encoded)  # Normalize to [-1, 1]
        
        quantum_logger.debug(f"QuantumLayer.forward: input shape {x.shape}, encoded {x_encoded.shape}")
        
        if PL_AVAILABLE:
            # Use PennyLane quantum circuit with configured backend
            output = self._quantum_forward(x_encoded)
        else:
            # Use simplified classical approximation
            output = self._classical_approximation(x_encoded)
        
        # Project back to output space
        output = self.fc_out(output)
        
        if self.use_bias:
            output = output + self.bias
        
        quantum_logger.debug(f"QuantumLayer output shape: {output.shape}")
        return output
    
    # Versão original do código, sem GPU Batching, que pode ser reativada futuramente para comparação de desempenho
    # def _quantum_forward(self, x_encoded: torch.Tensor) -> torch.Tensor:
    #     """Execute quantum circuit using PennyLane without GPU Batching (original version)."""
    #     try:
    #         import pennylane as qml
            
    #         batch_size = x_encoded.shape[0] if len(x_encoded.shape) > 1 else 1
            
    #         # Create quantum device and circuit
    #         dev = qml.device('default.qubit', wires=self.n_qubits)
            
    #         @qml.qnode(dev)
    #         def quantum_circuit(inputs, params):
    #             # Encoding layer: encode classical inputs
    #             for i in range(self.n_qubits):
    #                 qml.RY(np.pi * inputs[i], wires=i)
                
    #             # Variational quantum circuit
    #             for layer in range(self.n_layers):
    #                 # Single-qubit rotations
    #                 for i in range(self.n_qubits):
    #                     qml.RY(params[i, layer, 0], wires=i)
    #                     qml.RZ(params[i, layer, 1], wires=i)
                    
    #                 # Entangling layer: CNOT ladder
    #                 for i in range(self.n_qubits - 1):
    #                     qml.CNOT(wires=[i, i + 1])
    #                 qml.CNOT(wires=[self.n_qubits - 1, 0])  # Close ring
                
    #             # Measurement: expectation value of Z on each qubit
    #             return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
    #         # Execute circuit
    #         results = []
    #         for b in range(batch_size):
    #             output_b = quantum_circuit(
    #                 x_encoded[b] if len(x_encoded.shape) > 1 else x_encoded,
    #                 self.theta
    #             )
    #             results.append(torch.tensor(output_b, dtype=x_encoded.dtype, device=x_encoded.device))
            
    #         output = torch.stack(results)
    #         return output
            
    #     except Exception as e:
    #         quantum_logger.error(f"Quantum forward failed: {e}, falling back to classical")
    #         return self._classical_approximation(x_encoded)
    
    def _quantum_forward(self, x_encoded: torch.Tensor) -> torch.Tensor:
        """Execute quantum circuit using PennyLane with configured backend and diff_method."""
        try:
            import pennylane as qml
            import torch

            # Try configured backend with fallback chain
            dev = None
            actual_backend = self.backend
            actual_diff_method = self.diff_method

            for try_backend in [self.backend, 'lightning.qubit', 'default.qubit']:
                try:
                    dev = qml.device(try_backend, wires=self.n_qubits)
                    actual_backend = try_backend
                    break
                except Exception as e:
                    if try_backend == self.backend:
                        quantum_logger.warning(
                            f"Configured backend '{self.backend}' unavailable: {e}. "
                            f"Trying fallback..."
                        )
                    continue

            if dev is None:
                quantum_logger.error("All PennyLane backends failed, using classical approximation")
                return self._classical_approximation(x_encoded)

            # Re-validate diff_method for the actual backend
            if actual_backend != self.backend:
                actual_backend, actual_diff_method = _validate_backend_diff_method(
                    actual_backend, self.diff_method
                )

            # Log runtime info once per layer instance
            if not self._logged_runtime_info:
                quantum_logger.info(
                    f"QuantumLayer({self.in_features}→{self.out_features}): "
                    f"backend={actual_backend}, diff_method={actual_diff_method}"
                )
                _RUNTIME_QUANTUM_INFO.update({
                    'backend': actual_backend,
                    'diff_method': actual_diff_method,
                    'backend_requested': self.backend,
                    'diff_method_requested': self.diff_method,
                })
                self._logged_runtime_info = True

            _n_qubits = self.n_qubits
            _rotation = self.rotation
            _topology = self.topology

            @qml.qnode(dev, interface="torch", diff_method=actual_diff_method)
            def quantum_circuit(inputs, params):
                # Encoding: RY(π*x) per qubit (batched)
                for i in range(_n_qubits):
                    qml.RY(np.pi * inputs[:, i], wires=i)

                # VQC: PCE ansatz with configurable rotation/topology
                pce_ansatz(params, R_type=_rotation, n_qubits=_n_qubits, topology=_topology)

                return [qml.expval(qml.PauliZ(i)) for i in range(_n_qubits)]

            if len(x_encoded.shape) == 1:
                x_encoded = x_encoded.unsqueeze(0)

            chunk_size = 512
            output_chunks = []

            for i in range(0, x_encoded.shape[0], chunk_size):
                x_chunk = x_encoded[i:i + chunk_size]
                res_chunk = quantum_circuit(x_chunk, self.theta)
                output_chunk = torch.stack(res_chunk, dim=1)
                output_chunks.append(output_chunk)

            output = torch.cat(output_chunks, dim=0)
            output = output.to(dtype=x_encoded.dtype, device=x_encoded.device)
            return output

        except Exception as e:
            quantum_logger.error(f"Quantum forward failed: {e}")
            return self._classical_approximation(x_encoded)
        
    def _classical_approximation(self, x_encoded: torch.Tensor) -> torch.Tensor:
        """Classical approximation of quantum circuit for debugging/testing."""
        batch_shape = x_encoded.shape[:-1]

        x = x_encoded.unsqueeze(-1)  # (..., n_qubits, 1)

        for layer in range(self.n_layers):
            theta_layer = self.theta[layer]  # (n_qubits, n_rot_gates)

            # Use first two rotation params as RY/RZ-like weights
            weights = torch.sin(theta_layer[:, 0:1])
            if self.n_rot_gates > 1:
                weights = weights * torch.cos(theta_layer[:, 1:2])
            x = x * weights.unsqueeze(0)

            # Entanglement: mix adjacent qubits
            x_shifted = torch.roll(x, shifts=1, dims=-2)
            x = 0.7 * x + 0.3 * x_shifted

        output = x.squeeze(-1)
        return output


# Factory function for automatic backend selection
def create_quantum_layer(
    in_features: int,
    out_features: int,
    n_qubits: int = 4,
    n_layers: int = 4,
    use_bias: bool = True,
    rotation: str = 'RYRZ',
    topology: str = 'ring',
    backend: str = 'lightning.qubit',
    diff_method: str = 'best',
) -> nn.Module:
    """
    Create a quantum layer.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        n_qubits: Number of qubits
        n_layers: Number of VQC layers
        use_bias: Use bias in output projection
        rotation: Rotation gate sequence, e.g. 'RYRZ', 'RXRYRZ'
        topology: Entanglement topology — 'ring', 'brickwall', 'chain', 'lambda'
        backend: PennyLane device — 'lightning.gpu', 'lightning.qubit', etc.
        diff_method: Gradient method — 'best', 'adjoint', 'backprop', etc.
        
    Returns:
        QuantumLayer instance
    """
    quantum_logger.info(
        f"Creating QuantumLayer({in_features}, {out_features}, {n_qubits}, {n_layers}) "
        f"backend={backend}, diff_method={diff_method}"
    )
    return QuantumLayer(
        in_features, out_features, n_qubits, n_layers, use_bias,
        rotation=rotation, topology=topology,
        backend=backend, diff_method=diff_method,
    )


class SwitchableLinear(nn.Module):
    """Drop-in replacement for nn.Linear that optionally uses a quantum layer."""

    def __init__(self, in_features, out_features, use_quantum=False,
                 n_qubits=4, n_layers=4, rotation='RYRZ', topology='ring',
                 use_bias=True,
                 backend='lightning.qubit', diff_method='best'):
        super().__init__()
        if use_quantum:
            self.layer = create_quantum_layer(
                in_features, out_features, n_qubits=n_qubits, n_layers=n_layers,
                use_bias=use_bias,
                rotation=rotation, topology=topology,
                backend=backend, diff_method=diff_method,
            )
        else:
            self.layer = nn.Linear(in_features, out_features, bias=use_bias)
        self.is_quantum = use_quantum

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    # Test imports and backend detection
    print(f"PennyLane available: {PL_AVAILABLE}")
    
    # Create and test a layer
    layer = create_quantum_layer(128, 128, n_qubits=4, n_layers=4,
                                  backend='lightning.qubit', diff_method='best')
    x = torch.randn(32, 128)
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Runtime quantum info: {get_runtime_quantum_info()}")
    
    # Test backward pass
    loss = y.sum()
    loss.backward()
    print("Backward pass successful")
    print(f"Theta grad norm: {layer.theta.grad.norm().item():.6f}")
