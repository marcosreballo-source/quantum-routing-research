#!/usr/bin/env python3
"""
Batch SageMaker submitter for QGAT ablation studies.

Submits one SageMaker Training Job per YAML config found in ablation_yamls/{phase}/.
Writes a CSV job manifest to ablation_yamls/job_manifest.csv.

Usage
-----
    # Dry-run: preview jobs without submitting
    python scripts/submit_ablation_batch.py --phases 1A --dry-run

    # Submit Phase 1A only (3 concurrent slots)
    python scripts/submit_ablation_batch.py --phases 1A --max-concurrent 3

    # Submit multiple phases
    python scripts/submit_ablation_batch.py --phases 1A 1B

    # Cost estimate preview
    python scripts/submit_ablation_batch.py --phases 2 --dry-run --cost-estimate

    # Resume: skip already submitted experiment_ids (reads manifest)
    python scripts/submit_ablation_batch.py --phases 1A --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent          # baseline_merged/
ABLATION_DIR = BASE_DIR / "ablation_yamls"
MANIFEST_PATH = ABLATION_DIR / "job_manifest.csv"
LAUNCH_SCRIPT = BASE_DIR / "launch_sagemaker.py"

# Rough cost per hour by instance type (USD)
INSTANCE_COSTS = {
    "ml.g4dn.2xlarge": 0.74,
    "ml.g5.xlarge": 1.06,
    "ml.g5.2xlarge": 1.06,
    "ml.g5.4xlarge": 1.83,
}
DEFAULT_INSTANCE = "ml.g4dn.2xlarge"
ESTIMATED_HOURS_PER_JOB = 2.0  # conservative estimate for 100 epochs on 100k data


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

MANIFEST_FIELDS = [
    "experiment_id", "job_name", "phase", "yaml_file",
    "n_nodes", "data_size", "submitted_by", "submitted_time", "status",
]


def _read_manifest() -> dict[str, dict]:
    """Return {experiment_id: row} for all entries in the manifest."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return {row["experiment_id"]: row for row in reader}


def _append_manifest(rows: list[dict]) -> None:
    """Append rows to the manifest CSV (create with header if needed)."""
    write_header = not MANIFEST_PATH.exists()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# YAML discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_yamls(phases: list[str]) -> list[tuple[str, Path]]:
    """Return list of (phase, yaml_path) sorted by phase then filename."""
    results = []
    for phase in phases:
        phase_dir = ABLATION_DIR / phase
        if not phase_dir.is_dir():
            print(f"  [WARN] Phase directory not found: {phase_dir.relative_to(BASE_DIR)}")
            continue
        for yml in sorted(phase_dir.glob("*.yaml")):
            results.append((phase, yml))
    return results


def _load_experiment_id(yml_path: Path) -> str:
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    return data.get("experiment_id", yml_path.stem)


def _extract_metadata(yml_path: Path) -> dict:
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    return {
        "n_nodes": data.get("problem", {}).get("n_nodes", 20),
        "data_size": data.get("dataset", {}).get("data_size", "?"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Submission
# ─────────────────────────────────────────────────────────────────────────────

def submit_job(yml_path: Path, instance_type: str, dry_run: bool) -> str | None:
    """
    Call launch_sagemaker.py for a single YAML.
    Returns the job name (from stdout) or None on dry-run / failure.
    """
    cmd = [
        sys.executable, str(LAUNCH_SCRIPT),
        "--config", str(yml_path.relative_to(BASE_DIR)),
        "--instance", instance_type,
        "--no-wait",  # fire-and-forget, check status with check_and_download.py
    ]
    if dry_run:
        print(f"    [DRY-RUN] Would run: {' '.join(cmd)}")
        return f"DRY_RUN_{yml_path.stem}"

    print(f"    Launching: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        print(f"    [ERROR] launch_sagemaker.py failed:\n{result.stderr}")
        return None

    # Parse job name from launch_sagemaker.py stdout (it prints "Job name: ...")
    job_name = None
    for line in result.stdout.splitlines():
        if "job" in line.lower() and "name" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                job_name = parts[-1].strip()
                break
    if job_name is None:
        # Fallback: last non-empty line
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        job_name = lines[-1] if lines else "UNKNOWN"
    return job_name


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimate
# ─────────────────────────────────────────────────────────────────────────────

def print_cost_estimate(yaml_list: list[tuple[str, Path]], instance_type: str) -> None:
    cost_per_h = INSTANCE_COSTS.get(instance_type, 1.0)
    n = len(yaml_list)
    total_h = n * ESTIMATED_HOURS_PER_JOB
    total_cost = total_h * cost_per_h
    print(f"\n  Cost estimate ({instance_type} @ ${cost_per_h:.2f}/h):")
    print(f"    Jobs:          {n}")
    print(f"    Hours/job:     ~{ESTIMATED_HOURS_PER_JOB:.1f}h (estimate)")
    print(f"    Total hours:   ~{total_h:.0f}h")
    print(f"    Total cost:    ~${total_cost:.2f} USD")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

VALID_PHASES = ["1A", "1B", "1C", "2", "3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit ablation batch to SageMaker")
    parser.add_argument(
        "--phases", nargs="+", choices=VALID_PHASES, default=["1A"],
        help="Which phases to submit (default: 1A)",
    )
    parser.add_argument(
        "--instance", default=DEFAULT_INSTANCE,
        help=f"SageMaker instance type (default: {DEFAULT_INSTANCE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview jobs without submitting",
    )
    parser.add_argument(
        "--cost-estimate", action="store_true",
        help="Print cost estimate and exit",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiment_ids already in job_manifest.csv",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=0,
        help="Limit concurrent job submissions (0 = unlimited). Adds a short pause between batches.",
    )
    parser.add_argument(
        "--submitted-by", default=os.environ.get("USER", "unknown"),
        help="Name/alias to record in manifest (default: $USER)",
    )
    args = parser.parse_args()

    # Discover YAMLs
    yaml_list = discover_yamls(args.phases)
    if not yaml_list:
        print("No YAML files found. Run generate_ablation_yamls.py first.")
        sys.exit(1)

    print(f"Found {len(yaml_list)} YAML configs across phases: {args.phases}")

    if args.cost_estimate or args.dry_run:
        print_cost_estimate(yaml_list, args.instance)

    if args.cost_estimate and not args.dry_run:
        sys.exit(0)

    # Skip already submitted
    existing = _read_manifest() if args.skip_existing else {}
    skipped = 0

    submitted_rows: list[dict] = []
    concurrent_count = 0

    for phase, yml_path in yaml_list:
        eid = _load_experiment_id(yml_path)

        if eid in existing:
            print(f"  [SKIP] {eid} already in manifest")
            skipped += 1
            continue

        meta = _extract_metadata(yml_path)
        print(f"\n  [{phase}] {yml_path.name}  (experiment_id={eid})")

        job_name = submit_job(yml_path, args.instance, args.dry_run)
        status = "submitted" if (job_name and not args.dry_run) else ("dry_run" if args.dry_run else "failed")

        row = {
            "experiment_id": eid,
            "job_name": job_name or "",
            "phase": phase,
            "yaml_file": yml_path.name,
            "n_nodes": meta["n_nodes"],
            "data_size": meta["data_size"],
            "submitted_by": args.submitted_by,
            "submitted_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
        }
        submitted_rows.append(row)

        if not args.dry_run:
            concurrent_count += 1
            if args.max_concurrent > 0 and concurrent_count % args.max_concurrent == 0:
                print(f"\n  [{concurrent_count} jobs submitted — pausing 5s before next batch]")
                time.sleep(5)

    if submitted_rows and not args.dry_run:
        _append_manifest(submitted_rows)
        print(f"\nManifest updated: {MANIFEST_PATH.relative_to(BASE_DIR)}")

    print(f"\nSummary: {len(submitted_rows)} submitted, {skipped} skipped.")


if __name__ == "__main__":
    main()
