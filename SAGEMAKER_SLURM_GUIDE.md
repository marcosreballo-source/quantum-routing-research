# SageMaker & SLURM Guide — QGAT Merged

## Architecture Overview

```
configs/base.yaml + configs/experiment.yaml
         │
    main.py (unified entry point)
         │
    ┌────┴────┐
    │ TRAINERS │  ← training/__init__.py
    ├─────────┤
    │  'ppo'  │  → PPO_train.train()
    │'rollout'│  → VRP_Rollout_train.train()
    └─────────┘
```

Both SageMaker and SLURM invoke `main.py` with the same CLI interface.

---

## Local Development

```bash
# Activate environment
source .venv/bin/activate  # Linux
.venv\Scripts\Activate.ps1 # Windows

# Quick test
python main.py --base configs/base.yaml --config configs/light_test.yaml

# Classical mode (ignores quantum settings)
python main.py --base configs/base.yaml --config configs/light_test.yaml --mode classical

# Eval only
python main.py --base configs/base.yaml --config configs/baseline_10k.yaml \
    --checkpoint checkpoints/final.pt
```

---

## SLURM Deployment

### Submit a job

```bash
sbatch scripts/train.sbatch
```

### Standard SLURM script pattern

```bash
#!/bin/bash
#SBATCH --job-name=qgat_experiment
#SBATCH --partition=<PARTITION>
#SBATCH --output=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.out
#SBATCH --error=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.err

export MLFLOW_TRACKING_URI=http://servdpoa050:5001

uv run python main.py --config configs/your_experiment.yaml
```

### With scratch staging (recommended for large datasets)

```bash
#!/bin/bash
#SBATCH --job-name=qgat_512k

export TMP_DIR=/scratch/q-gat/runs/DRL_Lei/${SLURM_JOB_NAME}/
export RUN_DIR=/shared/q-gat/runs/DRL_Lei/${SLURM_JOB_NAME}/

uv run python main.py --config configs/baseline_512k.yaml

rsync -avH --progress $TMP_DIR $RUN_DIR
rm -rf $TMP_DIR
```

### Available scripts

| Script | Description |
|--------|-------------|
| `train.sbatch` | Basic PPO training (10k dataset) |
| `train_10k.sbatch` | 10k with 1k validation |
| `train_20k.sbatch` | 20k with CLI data overrides |
| `train_30k.sbatch` | 30k with scratch staging |
| `train_512k.sbatch` | Full dataset with staging |
| `train_10k_rol.sbatch` | Rollout baseline method |
| `train_768k_rol.sbatch` | Full rollout with staging |
| `eval_10k_ppo.sbatch` | Evaluation with checkpoint |
| `slurm_test.sbatch` | Quick test for CI |

### SLURM environment detection

`main.py` automatically detects SLURM via environment variables:
- `SLURM_JOB_ID` → logged as MLflow tag
- `SLURM_JOB_NAME` → logged as MLflow tag

---

## SageMaker Deployment

### Launch a training job

```bash
python launch_sagemaker.py \
    --config configs/baseline_512k.yaml \
    --instance ml.g4dn.2xlarge \
    --max-hours 48
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Experiment YAML |
| `--base-config` | `configs/base.yaml` | Base config |
| `--instance` | `ml.g4dn.2xlarge` | SageMaker instance type |
| `--max-hours` | 120 | Max runtime hours |
| `--data_size` | from YAML | Override training data size |
| `--val_size` | from YAML | Override validation size |
| `--role` | auto | IAM role ARN |
| `--no-download` | false | Skip artifact download |
| `--no-wait` | false | Don't wait for completion |

### How it works

1. `launch_sagemaker.py` validates config via `validate_config.py`
2. Creates PyTorch estimator with `sagemaker_entry.py` as entry point
3. Uploads full `src/` directory to S3
4. `sagemaker_entry.py` sets up environment and runs `main.py`
5. Checkpoints/MLflow artifacts saved to `SM_MODEL_DIR`
6. On completion, `model.tar.gz` downloaded and extracted locally
7. MLflow runs imported to local `./mlruns/`

### Instance recommendations

| Problem Size | Instance | Notes |
|-------------|----------|-------|
| n=20 classical | `ml.g4dn.xlarge` | 1x T4, 4 vCPU |
| n=20 quantum | `ml.g4dn.2xlarge` | 1x T4, 8 vCPU (CPU-bound quantum) |
| n=50 | `ml.g4dn.4xlarge` | 1x T4, 16 vCPU |
| n=100 | `ml.g4dn.8xlarge` | 1x T4, 32 vCPU |

---

## Config Override Priority

```
base.yaml → experiment.yaml → CLI overrides (--data_size, --mode)
```

The `apply_defaults()` function in `config.py` fills missing keys with safe defaults.

---

## New Features Available

### REINFORCE Baselines (Rollout training)

```yaml
training:
  method: rollout
  baseline: rollout      # rollout | exponential | critic | none
  bl_warmup_epochs: 1    # exponential warmup before rollout kicks in
```

### Beam Search Evaluation

```yaml
evaluation:
  beam_width: 10         # 0 = disabled (rollout models only)
```

### Data Distributions

```yaml
dataset:
  distribution: "cluster"  # uniform | cluster | grid | mixed
```

### Decoder Quantum Layers

```yaml
quantum:
  decoder:
    fc: true              # quantum context projection
    fc1: false            # quantum pool projection
```

### FFN + Normalization

```yaml
model:
  use_ffn: true
  normalization: "layer"   # batch | layer | none
```

### Ansatz Configuration

```yaml
quantum:
  rotation: "RXRYRZ"      # RY | RYRZ | RXRYRZ
  topology: "brickwall"   # ring | brickwall | chain | lambda
```

---

## Monitoring

### MLflow UI

```bash
# Local
mlflow ui --port 5001

# Access remote (SLURM server)
mlflow ui --backend-store-uri http://servdpoa050:5001
```

### Key metrics logged

| Metric | Description |
|--------|-------------|
| `train/mean_reward` | Batch-averaged tour cost |
| `train/mean_loss` | REINFORCE/PPO loss |
| `val/mean_cost` | Validation greedy cost |
| `eval/greedy_mean_cost` | Eval greedy cost |
| `eval/sampling_mean_cost` | Eval best-of-N sampling cost |
| `eval/beam_mean_cost` | Eval beam search cost |
| `rollout/baseline_mean` | Rollout baseline value |
| `opt/lr` | Current learning rate |
