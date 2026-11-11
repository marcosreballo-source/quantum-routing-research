#!/bin/bash

#SBATCH --job-name=baseline_qgat
#SBATCH --chdir=/shared/q-gat/projects/q-gat-tests
#SBATCH --output=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.out
#SBATCH --error=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.err
#SBATCH --partition=<PARTITION>

# MLflow points to head server
export MLFLOW_TRACKING_URI=http://servdpoa050:5001

uv run python main.py --config configs/baseline_10k.yaml
