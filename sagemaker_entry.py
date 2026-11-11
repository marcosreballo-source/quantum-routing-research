"""
Entry-point invocado pelo SageMaker Training Job.

O SageMaker define as variáveis de ambiente:
  SM_MODEL_DIR   -> /opt/ml/model
  SM_OUTPUT_DIR  -> /opt/ml/output
  SM_NUM_GPUS    -> número de GPUs disponíveis

Hiperparâmetros como argumentos CLI:
  --BASE_CONFIG, --CONFIG_FILE, --DATA_SIZE, --VAL_SIZE, --MODE
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def main():
    source_dir = Path(__file__).resolve().parent

    # Add src/ to PYTHONPATH
    src_dir = str(source_dir / "src")
    sys.path.insert(0, src_dir)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = src_dir + (":" + existing_pythonpath if existing_pythonpath else "")
    print(f"[sagemaker_entry] PYTHONPATH: {src_dir}")

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))

    # Parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--BASE_CONFIG", default=None)
    parser.add_argument("--CONFIG_FILE", default=None)
    parser.add_argument("--DATA_SIZE", default=None)
    parser.add_argument("--VAL_SIZE", default=None)
    parser.add_argument("--MODE", default=None)
    cli_args, _ = parser.parse_known_args()

    config_file = cli_args.CONFIG_FILE or os.environ.get("SM_HP_CONFIG_FILE", "configs/base.yaml")
    base_config = cli_args.BASE_CONFIG or os.environ.get("SM_HP_BASE_CONFIG", "configs/base.yaml")
    data_size = cli_args.DATA_SIZE or os.environ.get("SM_HP_DATA_SIZE", "")
    val_size = cli_args.VAL_SIZE or os.environ.get("SM_HP_VAL_SIZE", "")
    mode = cli_args.MODE or os.environ.get("SM_HP_MODE", "")

    config_file = config_file.replace("\\", "/")
    base_config = base_config.replace("\\", "/")
    print(f"[sagemaker_entry] base_config: {base_config}")
    print(f"[sagemaker_entry] config_file: {config_file}")

    base_path = source_dir / base_config
    config_path = source_dir / config_file

    if not base_path.exists():
        print(f"[sagemaker_entry] ERROR: Base config not found: {base_path}")
        configs_dir = source_dir / "configs"
        if configs_dir.exists():
            for f in sorted(configs_dir.iterdir()):
                print(f"  {f.name}")
        sys.exit(1)
    if not config_path.exists():
        print(f"[sagemaker_entry] ERROR: Config not found: {config_path}")
        sys.exit(1)

    # MLflow local file store
    mlruns_dir = source_dir / "mlruns"
    os.environ["MLFLOW_TRACKING_URI"] = mlruns_dir.as_uri()

    run_dir = output_dir / "runs"
    os.environ["RUN_DIR"] = str(run_dir)
    os.environ["RUN_NAME"] = "sagemaker"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Call main.py instead of train_PPO.py
    cmd = [
        sys.executable, "main.py",
        "--base", base_config,
        "--config", config_file,
    ]
    if data_size:
        cmd += ["--data_size", data_size]
    if val_size:
        cmd += ["--val_size", val_size]
    if mode:
        cmd += ["--mode", mode]

    print(f"[sagemaker_entry] CWD: {source_dir}")
    print(f"[sagemaker_entry] Command: {' '.join(cmd)}")
    print(f"[sagemaker_entry] MLflow: {mlruns_dir}")
    print(f"[sagemaker_entry] Run dir: {run_dir}")
    print(f"[sagemaker_entry] Model dir: {model_dir}")

    result = subprocess.run(cmd, cwd=str(source_dir))

    # Copy artifacts to SM_MODEL_DIR
    print("[sagemaker_entry] Copying artifacts to SM_MODEL_DIR...")
    _copy_tree(run_dir, model_dir / "runs")

    if mlruns_dir.exists():
        _copy_tree(mlruns_dir, model_dir / "mlruns")
        print(f"[sagemaker_entry] mlruns copied to {model_dir / 'mlruns'}")

    resolved = list(run_dir.rglob("config.resolved.yaml"))
    if resolved:
        shutil.copy2(resolved[0], model_dir / "config.resolved.yaml")

    print(f"[sagemaker_entry] Done (exit code: {result.returncode})")
    sys.exit(result.returncode)


def _copy_tree(src: Path, dst: Path):
    if src.exists():
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
