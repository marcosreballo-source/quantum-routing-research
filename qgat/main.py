from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import mlflow
import torch

from DRL_Lei.training import TRAINERS, REQUIRED_PARAMS
from DRL_Lei.utils.config import load_config, validate_params, namespace_to_dict, load_eval_config
from DRL_Lei.utils.mlflow_run import SafeMLflowRun, log_config_to_mlflow
from DRL_Lei.utils.paths import make_run_dir, make_eval_dir
from DRL_Lei.utils.architecture import describe_architecture, identify_strategy, extract_experiment_id
from DRL_Lei.utils.mlflow_setup import init_mlflow_tracking
from DRL_Lei.evaluate import evaluate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment config yaml"
    )
    parser.add_argument(
        "--base",
        default="configs/base.yaml",
        help="Base config (default: unified base.yaml)"
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "classical", "hybrid_cpu", "hybrid_gpu", "production"],
        default=None,
        help="Override config mode (sets device, verbosity, profiler, quantum)"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        help="Number of samples to generate for training"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        help="Number of samples to generate for validation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt file for standalone evaluation (skips training)"
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Optional yaml to override evaluation parameters only"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking (useful for local runs)"
    )

    args = parser.parse_args()

    # ---------- Load config ----------
    cfg = load_config(Path(args.base), Path(args.config))

    # CLI mode override
    if args.mode is not None:
        cfg.mode = args.mode
        from DRL_Lei.utils.config import resolve_mode
        resolve_mode(cfg)

    if args.data_size is not None:
        cfg.dataset.data_size = args.data_size
    if args.val_size is not None:
        cfg.dataset.val_size = args.val_size
    if args.eval_config is not None:
        cfg = load_eval_config(cfg, args.eval_config)

    device = getattr(getattr(cfg, 'runtime', None), 'device', None)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ---------- Check method params ----------
    method = cfg.training.method
    if method not in TRAINERS:
        print(f"Unsupported training method: {method}")
        sys.exit(1)
    if method in REQUIRED_PARAMS:
        validate_params(vars(cfg.training), REQUIRED_PARAMS[method])

    # ---------- Extract experiment_id ----------
    resolved_cfg = namespace_to_dict(cfg)
    experiment_id = extract_experiment_id(resolved_cfg, config_path=args.config)
    print(f'Experiment ID: {experiment_id}')

    # ---------- Create run directory ----------
    run_dir = make_run_dir(cfg, experiment_id=experiment_id) if args.checkpoint is None else None

    # Save resolved config
    resolved_cfg_path = (run_dir / "artifacts" / "config.resolved.yaml"
                         if run_dir is not None
                         else args.checkpoint.parent / "config.resolved.yaml")
    with open(resolved_cfg_path, "w") as f:
        yaml.safe_dump(resolved_cfg, f)

    def _run():
        if args.checkpoint is not None:
            eval_dir = make_eval_dir(args.checkpoint)
            if mlflow.active_run() is not None:
                mlflow.set_tag("checkpoint", str(args.checkpoint))
            evaluate(cfg, args.checkpoint, eval_dir)
        else:
            TRAINERS[method](cfg, run_dir)

    if args.no_mlflow:
        _run()
    else:
        # Initialize MLflow tracking backend (SQLite/PostgreSQL/default)
        init_mlflow_tracking(resolved_cfg)

        # Experiment name includes phase context from experiment_id
        phase = experiment_id.split('_')[0] if '_' in experiment_id else 'default'
        mlflow_prefix = resolved_cfg.get('mlflow', {}).get('experiment_prefix', 'DRL_Lei')
        mlflow.set_experiment(f"{mlflow_prefix}/vrp{cfg.problem.n_nodes}/{phase}")
        run_name = run_dir.name if run_dir is not None else f"eval_{args.checkpoint.parents[1].name}"
        with SafeMLflowRun(run_name=run_name):
            if run_dir is not None:
                mlflow.set_tag("run_dir", str(run_dir))
            if os.environ.get("SLURM_JOB_ID") is not None:
                mlflow.set_tag("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
                mlflow.set_tag("slurm_job_name", os.environ.get("SLURM_JOB_NAME"))

            # Quantum tags + architecture description
            q = getattr(cfg, 'quantum', None)
            if q and getattr(q, 'enabled', False):
                mlflow.set_tag("quantum.enabled", "true")
                mlflow.set_tag("quantum.backend", getattr(q, 'backend', 'lightning.qubit'))
                mlflow.set_tag("quantum.diff_method", getattr(q, 'diff_method', 'best'))
            else:
                mlflow.set_tag("quantum.enabled", "false")

            arch_desc = describe_architecture(resolved_cfg)
            print(arch_desc)
            mlflow.log_text(arch_desc, "architecture_summary.txt")
            mlflow.set_tag("architecture_strategy", identify_strategy(resolved_cfg))
            mlflow.set_tag("experiment_id", experiment_id)

            mlflow.set_tag("mode", getattr(cfg, 'mode', 'unknown'))
            log_config_to_mlflow(resolved_cfg)
            _run()

            # Log actual runtime quantum backend/diff_method (populated after first forward pass)
            try:
                from DRL_Lei.quantum_layers import get_runtime_quantum_info
                rt_info = get_runtime_quantum_info()
                if rt_info:
                    mlflow.set_tag("quantum.backend_actual", rt_info.get('backend', ''))
                    mlflow.set_tag("quantum.diff_method_actual", rt_info.get('diff_method', ''))
            except Exception:
                pass  # quantum_layers may not have run if quantum is disabled


if __name__ == "__main__":
    main()
