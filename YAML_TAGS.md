# YAML Configuration Reference — QGAT Merged

All configuration is done through YAML files under `configs/`.  
Experiment configs inherit from `base.yaml` and override specific keys.

## Quick Start

```bash
# Local training
python main.py --base configs/base.yaml --config configs/baseline_10k.yaml

# SLURM
sbatch scripts/train.sbatch

# SageMaker
python launch_sagemaker.py --config configs/baseline_512k.yaml
```

---

## Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | str | `hybrid_gpu` | Execution mode: `debug`, `classical`, `hybrid_cpu`, `hybrid_gpu`, `production` |

---

## `problem`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_nodes` | int | 20 | Number of customer nodes (depot is added automatically) |

---

## `dataset`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_size` | int | 512000 | Number of training instances |
| `val_size` | int | 10000 | Number of validation instances |
| `distribution` | str | `"uniform"` | Node coordinate distribution: `uniform`, `cluster`, `grid`, `mixed` |

---

## `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hidden_node_dim` | int | 128 | Node embedding dimension |
| `hidden_edge_dim` | int | 16 | Edge embedding dimension |
| `conv_layers` | int | 4 | Number of GAT encoder layers |
| `use_ffn` | bool | `false` | Add FFN (Linear→4x→ReLU→Linear) after each GAT layer |
| `normalization` | str | `"batch"` | Normalization after GAT/FFN: `batch`, `layer`, `none` |
| `tanh_clipping` | float | 10.0 | Logit clipping magnitude in decoder attention |

---

## `training`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `"ppo"` | Training algorithm: `ppo`, `rollout` |
| `policy_mode` | str | `"sampling"` | PPO policy: `sampling`, `greedy` |
| `lr` | float | 3e-4 | Learning rate |
| `batch_size` | int | 512 | Batch size |
| `epochs` | int | 100 | Number of training epochs |
| `lr_decay` | float | 1.0 | LR decay per epoch (LambdaLR) |
| `baseline` | str\|null | `null` | REINFORCE baseline: `rollout`, `exponential`, `critic`, `none`, `null` (auto) |
| `bl_warmup_epochs` | int | 1 | Warmup epochs for rollout baseline (exponential during warmup) |
| `rollout_baseline_quantum` | bool | `true` | Use quantum layers in rollout baseline model |
| `timestep` | int | 1 | PPO timesteps per rollout |
| `ppo_epoch` | int | 3 | PPO update epochs per timestep |
| `entropy_value` | float | 0.01 | PPO entropy bonus coefficient |
| `eps_clip` | float | 0.2 | PPO clipping epsilon |

---

## `evaluation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | 1280 | Number of sampling draws per instance |
| `eval_size` | int | 1000 | Number of eval instances |
| `beam_width` | int | 0 | Beam search width (0 = disabled, rollout models only) |
| `temperatures` | dict | `{21: 2.5, 51: 1.8, 101: 1.2}` | Temperature per problem size (key = n_nodes+1) |

---

## `quantum`

Master switch and per-component quantum layer selection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Master quantum switch |
| `rotation` | str | `"RYRZ"` | Gate sequence per qubit: `"RY"`, `"RYRZ"`, `"RXRYRZ"` |
| `topology` | str | `"ring"` | Entanglement pattern: `ring`, `brickwall`, `chain`, `lambda` |
| `n_qubits_actor` | int | 4 | Qubits for actor quantum layers |
| `n_qubits_critic` | int | 4 | Qubits for critic quantum layers |
| `n_layers_quantum_actor` | int | 4 | Variational layers for actor |
| `n_layers_quantum_critic` | int | 2 | Variational layers for critic |
| `backend` | str | `"lightning.qubit"` | PennyLane device backend: `lightning.gpu`, `lightning.qubit`, `lightning.kokkos`, `default.qubit` |
| `diff_method` | str | `"best"` | Gradient differentiation method: `best` (auto-select), `adjoint`, `backprop`, `parameter-shift`, `finite-diff` |

### `quantum.encoder`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fc_node` | bool | `false` | Quantum node projection linear |
| `fc_edge` | bool | `false` | Quantum edge projection linear |
| `gat_layers` | list[int] | `[]` | GAT layer indices with quantum fc/attn (e.g. `[0, 1, 2]`) |

### `quantum.decoder`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fc` | bool | `false` | Quantum context projection in decoder |
| `fc1` | bool | `false` | Quantum pool projection in decoder |
| `n_qubits` | int | 4 | Qubits for decoder quantum layers |
| `n_layers` | int | 4 | Variational layers for decoder |

### `quantum.critic`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable quantum critic |
| `layers` | list[int] | `[]` | Critic FC layer indices to make quantum |

---

## `temperature`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `initial` | float | 2.5 | Starting temperature for sampling |
| `final` | float | 2.5 | Final temperature |
| `schedule` | str | `"constant"` | Annealing: `constant`, `linear`, `exponential` |
| `decay_rate` | float | 1.0 | Decay rate for exponential schedule |

---

## `dropout`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gat` | float | 0.0 | Dropout in GAT layers |
| `attn` | float | 0.0 | Attention dropout |
| `attn_output` | float | 0.0 | Post-attention dropout |
| `decoder` | float | 0.0 | Decoder probability dropout |
| `critic` | float | 0.0 | Critic dropout |

---

## `seeds`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_seed` | int | 42 | Data generation seed |
| `model_seed` | int | 42 | Model initialization seed |
| `random_seed` | int | 42 | Python random seed |
| `dropout_seed` | int | 42 | Dropout seed |
| `sampling_seed` | int | 42 | Sampling seed |

---

## `hardware`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_gpu` | str | `"auto"` | GPU usage: `auto`, `true`, `false` |
| `num_workers` | str | `"auto"` | DataLoader workers |
| `pin_memory` | str | `"auto"` | Pin memory for GPU |

---

## `mlflow`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracking_uri` | str\|null | `null` | MLflow tracking URI (null = env var) |
| `experiment_prefix` | str | `"DRL_Lei"` | MLflow experiment name prefix |

---

## Ablation Config Examples

```yaml
# ablation_ffn.yaml — Test FFN with layer normalization
model:
  use_ffn: true
  normalization: "layer"

# ablation_ansatz.yaml — Test brickwall + RXRYRZ
quantum:
  enabled: true
  rotation: "RXRYRZ"
  topology: "brickwall"

# ablation_distribution.yaml — Train on clustered data
dataset:
  distribution: "cluster"

# ablation_beam.yaml — Enable beam search evaluation
evaluation:
  beam_width: 10
```
