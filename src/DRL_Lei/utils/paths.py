import os
from pathlib import Path
from datetime import datetime

# ==========================================================
# RUN DIRECTORY
# ==========================================================

# Project root is four levels up from this file:
# paths.py -> utils/ -> DRL_Lei/ -> src/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

def make_run_dir(cfg, experiment_id=None) -> Path:
    job_name = os.environ.get("SLURM_JOB_NAME")
    suffix = get_path_suffix()

    # Build semantic run name: {method}_{experiment_id}_{date}_{time}[_{slurm_job_id}]
    method = cfg.training.method
    if experiment_id:
        run_name = f"{method}_{experiment_id}_{suffix}"
    else:
        run_name = f"{method}_{suffix}"

    # Priority: TMP_DIR (slurm scratch) > SLURM shared > local runs/
    env_tmp_dir = os.environ.get("TMP_DIR")
    slurm_shared = Path("/shared/q-gat/runs/DRL_Lei")

    if env_tmp_dir is not None:
        base = Path(env_tmp_dir)
    elif job_name is not None:
        base = slurm_shared / job_name
    else:
        base = _PROJECT_ROOT / "runs"

    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "checkpoints").mkdir()
    (run_dir / "artifacts").mkdir()

    return run_dir


def make_eval_dir(checkpoint_path: Path) -> Path:
    """Creates an eval/ directory next to the checkpoint file."""
    eval_dir = checkpoint_path.parent / "eval"
    eval_dir.mkdir(exist_ok=True)

    return eval_dir


def get_path_suffix() -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = f"{date}_{time}" + (f"_{job_id}" if job_id else "")
    return suffix
