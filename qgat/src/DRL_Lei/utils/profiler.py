"""
Shared profiling utilities for all training and evaluation paths.

Provides:
  BlockTimer       — wall-clock timing of named code blocks (CUDA-sync aware)
  _NullTimer       — no-op drop-in replacement when profiling is disabled
  MemoryTracker    — peak GPU/CPU memory tracking
  log_device_info  — print device + VRAM summary
  log_eval_summary — print profiler results to stdout + MLflow
"""

import time
import logging
from collections import defaultdict
from contextlib import contextmanager

import torch

logger = logging.getLogger("DRL_Lei.profiler")

# Verbose level constants (matches config.py MODE_PRESETS)
V_NONE = 0
V_BASIC = 1
V_MEDIUM = 2
V_FULL = 3


# ---------------------------------------------------------------------------
# BlockTimer
# ---------------------------------------------------------------------------
class BlockTimer:
    """Wall-clock timer for named blocks with CUDA synchronization."""

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, block_name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        self.totals[block_name] += elapsed
        self.counts[block_name] += 1

    def reset(self):
        self.totals.clear()
        self.counts.clear()

    def summary(self, prefix: str = "") -> str:
        """Return formatted summary of all measured blocks."""
        if not self.totals:
            return f"{prefix}(no measurements)"
        total = sum(self.totals.values())
        lines = [f"{prefix}=== PROFILER SUMMARY ==="]
        for name in sorted(self.totals, key=lambda k: self.totals[k], reverse=True):
            t = self.totals[name]
            pct = (t / total * 100) if total > 0 else 0
            cnt = self.counts[name]
            avg = t / cnt if cnt > 0 else 0
            lines.append(f"{prefix}  {name}: {t:.3f}s ({pct:.1f}%) [{cnt} calls, avg {avg:.4f}s]")
        lines.append(f"{prefix}  TOTAL: {total:.3f}s")
        lines.append(f"{prefix}========================")
        return "\n".join(lines)


class _NullTimer:
    """No-op replacement when profiling is disabled."""

    @contextmanager
    def measure(self, block_name: str):
        yield

    def reset(self):
        pass

    @property
    def totals(self):
        return {}

    @property
    def counts(self):
        return {}

    def summary(self, prefix: str = "") -> str:
        return ""


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------
class MemoryTracker:
    """Track peak GPU memory during a scope."""

    def __init__(self):
        self._snapshots: dict[str, float] = {}  # MB

    def snapshot(self, label: str):
        if torch.cuda.is_available():
            self._snapshots[label] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def summary(self) -> str:
        if not self._snapshots:
            return ""
        lines = ["Memory snapshots (peak allocated):"]
        for label, mb in self._snapshots.items():
            lines.append(f"  {label}: {mb:.1f} MB")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def log_device_info(device: torch.device, verbose: int = V_BASIC):
    """Print device information at start of training/eval."""
    if verbose < V_BASIC:
        return
    info = f"Device: {device}"
    if device.type == "cuda":
        idx = device.index or 0
        info += f" ({torch.cuda.get_device_name(idx)})"
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        info += f", VRAM: {vram:.1f} GB"
        info += f", CUDA {torch.version.cuda}"
    print(info)


def log_quantum_summary(cfg, verbose: int = V_BASIC):
    """Print quantum configuration at start of training/eval."""
    if verbose < V_BASIC:
        return
    q = getattr(cfg, 'quantum', None)
    if q is None or not getattr(q, 'enabled', False):
        print("Quantum: DISABLED (classical mode)")
        return
    backend = getattr(q, 'backend', 'lightning.qubit')
    diff_method = getattr(q, 'diff_method', 'best')
    rotation = getattr(q, 'rotation', 'RYRZ')
    topology = getattr(q, 'topology', 'ring')
    nq_actor = getattr(q, 'n_qubits_actor', 4)
    nq_critic = getattr(q, 'n_qubits_critic', 4)
    print(f"Quantum: backend={backend}, diff_method={diff_method}, "
          f"rotation={rotation}, topology={topology}, "
          f"qubits(actor={nq_actor}, critic={nq_critic})")


def log_profiler_to_mlflow(profiler: BlockTimer, step: int, prefix: str = "profiler"):
    """Log profiler totals and percentages to MLflow."""
    try:
        import mlflow
        if mlflow.active_run() is None or not profiler.totals:
            return
        total = sum(profiler.totals.values())
        for bname, btime in profiler.totals.items():
            mlflow.log_metric(f"{prefix}_sec/{bname}", btime, step=step)
            if total > 0:
                mlflow.log_metric(f"{prefix}_pct/{bname}", (btime / total) * 100, step=step)
        mlflow.log_metric(f"{prefix}_sec/_total", total, step=step)
    except Exception:
        pass


def make_profiler(profiler_enabled: bool) -> BlockTimer | _NullTimer:
    """Create a BlockTimer or _NullTimer based on config."""
    return BlockTimer() if profiler_enabled else _NullTimer()
