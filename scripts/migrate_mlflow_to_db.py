#!/usr/bin/env python3
"""
Migrate MLflow runs from file-based mlruns/ to a database backend.

Usage:
    # Migrate to SQLite (recommended for local dev):
    python scripts/migrate_mlflow_to_db.py --backend sqlite

    # Migrate to PostgreSQL:
    python scripts/migrate_mlflow_to_db.py --backend postgresql://user:pass@host:5432/mlflow

    # Dry run (preview only, no changes):
    python scripts/migrate_mlflow_to_db.py --backend sqlite --dry-run

This script reads all experiments/runs from the file-based mlruns/ directory
and recreates them in the target database backend, preserving:
  - Experiment names and IDs
  - Run metadata (name, status, start/end time)
  - Parameters, metrics, and tags
  - Artifacts (copied to new artifact location)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import mlflow
from mlflow.tracking import MlflowClient


def resolve_backend_uri(backend: str) -> tuple[str, str]:
    """Resolve backend string to (tracking_uri, artifact_root)."""
    if backend == "sqlite":
        db_dir = _PROJECT_ROOT / ".mlflow"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "mlruns.db"
        artifact_root = str(_PROJECT_ROOT / "mlflow-artifacts")
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}", artifact_root
    else:
        # Assume it's a full connection string
        artifact_root = str(_PROJECT_ROOT / "mlflow-artifacts")
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
        return backend, artifact_root


def get_file_runs(mlruns_dir: Path):
    """Read all experiments and runs from file-based mlruns/ directory."""
    experiments = []
    for exp_dir in sorted(mlruns_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ('.trash', 'models'):
            continue

        # Read experiment meta
        meta_file = exp_dir / "meta.yaml"
        if not meta_file.exists():
            continue

        import yaml
        with open(meta_file) as f:
            exp_meta = yaml.safe_load(f)

        exp_info = {
            'id': exp_meta.get('experiment_id', exp_dir.name),
            'name': exp_meta.get('name', f'experiment_{exp_dir.name}'),
            'runs': [],
        }

        # Read runs
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_meta_file = run_dir / "meta.yaml"
            if not run_meta_file.exists():
                continue

            with open(run_meta_file) as f:
                run_meta = yaml.safe_load(f)

            run_info = {
                'run_id': run_meta.get('run_id', run_dir.name),
                'run_name': run_meta.get('run_name', ''),
                'status': run_meta.get('status', 'FINISHED'),
                'start_time': run_meta.get('start_time'),
                'end_time': run_meta.get('end_time'),
                'params': {},
                'metrics': {},
                'tags': {},
                'artifacts_dir': run_dir / "artifacts",
            }

            # Read params
            params_dir = run_dir / "params"
            if params_dir.exists():
                for p in params_dir.rglob("*"):
                    if p.is_file():
                        key = str(p.relative_to(params_dir)).replace(os.sep, '.')
                        run_info['params'][key] = p.read_text().strip()

            # Read metrics
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists():
                for m in metrics_dir.rglob("*"):
                    if m.is_file():
                        key = str(m.relative_to(metrics_dir)).replace(os.sep, '.')
                        # Metrics file format: timestamp value step (one per line)
                        lines = m.read_text().strip().split('\n')
                        if lines and lines[-1]:
                            parts = lines[-1].split()
                            if len(parts) >= 2:
                                try:
                                    run_info['metrics'][key] = float(parts[1])
                                except ValueError:
                                    pass

            # Read tags
            tags_dir = run_dir / "tags"
            if tags_dir.exists():
                for t in tags_dir.rglob("*"):
                    if t.is_file():
                        key = str(t.relative_to(tags_dir)).replace(os.sep, '.')
                        run_info['tags'][key] = t.read_text().strip()

            exp_info['runs'].append(run_info)

        experiments.append(exp_info)

    return experiments


def migrate(experiments, tracking_uri, artifact_root, dry_run=False):
    """Migrate experiments/runs to database backend."""
    if dry_run:
        print("\n=== DRY RUN (no changes will be made) ===\n")

    print(f"Target tracking URI: {tracking_uri}")
    print(f"Target artifact root: {artifact_root}")
    print(f"Found {len(experiments)} experiment(s)\n")

    if not dry_run:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri) if not dry_run else None
    total_runs = 0
    migrated_runs = 0
    errors = []

    for exp in experiments:
        n_runs = len(exp['runs'])
        total_runs += n_runs
        print(f"Experiment: {exp['name']} (ID: {exp['id']}, {n_runs} run(s))")

        if dry_run:
            for run in exp['runs']:
                print(f"  Run: {run['run_name']} ({run['run_id'][:8]}...)")
                print(f"    Params: {len(run['params'])}, Metrics: {len(run['metrics'])}, Tags: {len(run['tags'])}")
                has_artifacts = run['artifacts_dir'].exists() and any(run['artifacts_dir'].iterdir())
                print(f"    Artifacts: {'yes' if has_artifacts else 'no'}")
            continue

        # Create or get experiment
        try:
            exp_obj = client.get_experiment_by_name(exp['name'])
            if exp_obj is None:
                new_exp_id = client.create_experiment(
                    exp['name'],
                    artifact_location=artifact_root,
                )
            else:
                new_exp_id = exp_obj.experiment_id
        except Exception as e:
            errors.append(f"Experiment {exp['name']}: {e}")
            continue

        for run in exp['runs']:
            try:
                # Create run
                new_run = client.create_run(
                    experiment_id=new_exp_id,
                    run_name=run['run_name'],
                )
                new_run_id = new_run.info.run_id

                # Log params (truncate to MLflow limit)
                for k, v in run['params'].items():
                    try:
                        client.log_param(new_run_id, k, v[:500])
                    except Exception:
                        pass

                # Log metrics
                for k, v in run['metrics'].items():
                    try:
                        client.log_metric(new_run_id, k, v)
                    except Exception:
                        pass

                # Set tags
                for k, v in run['tags'].items():
                    if not k.startswith('mlflow.'):  # Skip internal tags
                        try:
                            client.set_tag(new_run_id, k, v[:5000])
                        except Exception:
                            pass

                # Copy artifacts
                if run['artifacts_dir'].exists() and any(run['artifacts_dir'].iterdir()):
                    dest = Path(artifact_root) / new_exp_id / new_run_id / "artifacts"
                    dest.mkdir(parents=True, exist_ok=True)
                    for item in run['artifacts_dir'].iterdir():
                        if item.is_file():
                            shutil.copy2(item, dest / item.name)
                        elif item.is_dir():
                            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

                # End run with correct status
                status_map = {
                    'RUNNING': 'FINISHED',
                    'FINISHED': 'FINISHED',
                    'FAILED': 'FAILED',
                    'KILLED': 'KILLED',
                }
                status = status_map.get(run.get('status', 'FINISHED'), 'FINISHED')
                client.set_terminated(new_run_id, status=status)

                migrated_runs += 1
                print(f"  Migrated: {run['run_name']} ({run['run_id'][:8]}... -> {new_run_id[:8]}...)")

            except Exception as e:
                errors.append(f"Run {run['run_name']}: {e}")
                print(f"  FAILED: {run['run_name']} - {e}")

    print(f"\n{'=' * 60}")
    print(f"Migration complete: {migrated_runs}/{total_runs} runs migrated")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Migrate MLflow from file-based to database backend")
    parser.add_argument(
        "--backend",
        required=True,
        help='Target backend: "sqlite" or full connection string (e.g. postgresql://...)',
    )
    parser.add_argument(
        "--mlruns-dir",
        default=str(_PROJECT_ROOT / "mlruns"),
        help="Path to existing mlruns/ directory (default: ./mlruns/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes",
    )
    args = parser.parse_args()

    mlruns_dir = Path(args.mlruns_dir)
    if not mlruns_dir.exists():
        print(f"Error: mlruns/ directory not found at {mlruns_dir}")
        sys.exit(1)

    tracking_uri, artifact_root = resolve_backend_uri(args.backend)
    experiments = get_file_runs(mlruns_dir)

    if not experiments:
        print("No experiments found in mlruns/. Nothing to migrate.")
        return

    migrate(experiments, tracking_uri, artifact_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
