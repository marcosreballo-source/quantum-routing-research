import os
import sys
import signal
import mlflow


class SafeMLflowRun:
    def __init__(self, run_name):
        self.run_name = run_name
        self.run = None
        self.terminated = False

    def _set_terminated(self, status):
        if self.run is not None and not self.terminated:
            try:
                mlflow.end_run(status=status)
            finally:
                self.terminated = True

    def _handle_signal(self, signum, frame):
        self._set_terminated("KILLED")
        sys.exit(128 + signum)

    def __enter__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        if os.environ.get("SLURM_JOB_ID") is not None:
            signal.signal(signal.SIGTERM, self._handle_signal)
        self.run = mlflow.start_run(run_name=self.run_name)
        return self.run

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._set_terminated("FINISHED")
        elif issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            self._set_terminated("KILLED")
        else:
            self._set_terminated("FAILED")
        return False

# ==========================================================
# MLflow utilities
# ==========================================================

def flatten_dict(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def log_config_to_mlflow(cfg_dict: dict):
    mlflow.log_params(flatten_dict(cfg_dict))
