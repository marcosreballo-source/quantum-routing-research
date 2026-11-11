"""
MLflow tracking backend initialization.

Supports three modes:
  1. File-based (legacy): mlruns/ directory — default when backend_store is null
  2. SQLite (local dev):  .mlflow/mlruns.db — set backend_store: "sqlite"
  3. PostgreSQL (cloud):  backend_store: "postgresql://user:pass@host:5432/mlflow"

Usage in main.py:
    init_mlflow_tracking(resolved_cfg)
"""
import os
import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def init_mlflow_tracking(cfg_dict: dict) -> str:
    """Initialize MLflow tracking URI from config.

    Args:
        cfg_dict: Resolved config as plain dict.

    Returns:
        The tracking URI that was set.
    """
    mlflow_cfg = cfg_dict.get('mlflow', {})

    # 1. Explicit tracking_uri takes precedence (or env var)
    tracking_uri = mlflow_cfg.get('tracking_uri')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI (explicit): %s", tracking_uri)
        return tracking_uri

    # 2. Check for env var
    env_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if env_uri:
        logger.info("MLflow tracking URI (env): %s", env_uri)
        return env_uri

    # 3. Determine backend from config
    backend_store = mlflow_cfg.get('backend_store')
    artifact_root = mlflow_cfg.get('artifact_root')

    if backend_store is None:
        # Legacy file-based mode — no changes needed, MLflow defaults to ./mlruns/
        logger.info("MLflow tracking: file-based (mlruns/)")
        return str(_PROJECT_ROOT / "mlruns")

    if backend_store == "sqlite":
        # SQLite local database
        db_dir = _PROJECT_ROOT / ".mlflow"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "mlruns.db"
        uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(uri)

        # Set artifact root
        if artifact_root is None:
            artifact_root = str(_PROJECT_ROOT / "mlflow-artifacts")
        os.environ.setdefault("MLFLOW_DEFAULT_ARTIFACT_ROOT", str(artifact_root))
        Path(artifact_root).mkdir(parents=True, exist_ok=True)

        logger.info("MLflow tracking: SQLite (%s)", db_path)
        logger.info("MLflow artifacts: %s", artifact_root)
        return uri

    if backend_store.startswith(("postgresql://", "mysql://", "mssql://")):
        # Remote database
        mlflow.set_tracking_uri(backend_store)

        if artifact_root:
            os.environ.setdefault("MLFLOW_DEFAULT_ARTIFACT_ROOT", str(artifact_root))

        logger.info("MLflow tracking: database (%s)", backend_store.split('@')[0] + '@***')
        return backend_store

    # Unknown backend — treat as raw URI
    mlflow.set_tracking_uri(backend_store)
    logger.warning("MLflow tracking: unknown backend type '%s'", backend_store)
    return backend_store
