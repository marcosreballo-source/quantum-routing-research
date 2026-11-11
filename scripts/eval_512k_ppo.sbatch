#!/bin/bash

#SBATCH --job-name=baseline_DRL_Lei_full
#SBATCH --chdir=/shared/q-gat/projects/q-gat-tests
#SBATCH --output=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.out
#SBATCH --error=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.err
#SBATCH --partition=<PARTITION>

# MLflow points to head server
export MLFLOW_TRACKING_URI=http://servdpoa050:5001

uv run python main.py \
    --config configs/baseline_512k.yaml \
    --checkpoint /shared/q-gat/runs/DRL_Lei/baseline_DRL_Lei_512k/ppo_2026-03-16_16-41-02_34/checkpoints/final.pt \
    --eval-config configs/eval.yaml
