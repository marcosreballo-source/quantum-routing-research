#!/bin/bash

#SBATCH --job-name=baseline_DRL_Lei_full
#SBATCH --chdir=/shared/q-gat/projects/q-gat-tests
#SBATCH --output=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.out
#SBATCH --error=/shared/q-gat/runs/DRL_Lei/%x/slurm/%x-%j.err
#SBATCH --partition=<PARTITION>

# MLflow points to head server
export MLFLOW_TRACKING_URI=http://servdpoa050:5001

export TMP_DIR=/scratch/q-gat/runs/DRL_Lei/${SLURM_JOB_NAME}/
export RUN_DIR=/shared/q-gat/runs/DRL_Lei/${SLURM_JOB_NAME}/

uv run python main.py --config configs/baseline_512k.yaml

rsync -avH --progress $TMP_DIR $RUN_DIR
rm -rf $TMP_DIR
