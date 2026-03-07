#!/usr/bin/env python3
"""run_icd9_pipeline.py — ICD-9-CM mapping pipeline runner.

Steps:
  1. Build ICD-9-CM knowledge base: read icd_9_dictionary.csv → embed → LanceDB
  2. Map uncoded problems: read mock_uncoded.csv → RAG → icd9_mapping_results.csv

Usage:
    python run_icd9_pipeline.py
    python run_icd9_pipeline.py --model danube3_500m
    python run_icd9_pipeline.py --output results/my_mapping.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_step(label: str, cmd: list[str]) -> int:
    log(f"{'─' * 60}")
    log(f"STEP: {label}")
    log(f"CMD:  {' '.join(cmd)}")
    log(f"{'─' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log(f"STEP FAILED (exit {result.returncode}): {label}")
    else:
        log(f"STEP OK: {label}")
    return result.returncode


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    _default_output = str(script_dir / "icd9_mapping_results.csv")

    parser = argparse.ArgumentParser(
        description="Full ICD-9-CM pipeline: flush LanceDB → build KB → map problems → CSV"
    )
    parser.add_argument("--model", metavar="PROFILE",
                        help="Model profile from model_config.yaml (overrides active_profile)")
    parser.add_argument("--output", metavar="PATH",
                        default=_default_output,
                        help=f"Output CSV path (default: {_default_output})")
    args = parser.parse_args()

    t0 = time.perf_counter()
    log("=" * 60)
    log("ICD-9-CM PIPELINE START")
    log("=" * 60)

    python = sys.executable

    # Step 1 — Flush LanceDB + build ICD-9 KB
    rc = run_step(
        "Flush LanceDB + embed ICD-9-CM dictionary → LanceDB",
        [python, str(script_dir / "03_build_icd9_kb.py")],
    )
    if rc != 0:
        log("Aborting pipeline — KB build failed.")
        return rc

    # Step 2 — Map uncoded problems → CSV
    map_cmd = [
        python, str(script_dir / "04_map_uncoded_problems.py"),
        "--output", args.output,
    ]
    if args.model:
        map_cmd += ["--model", args.model]

    rc = run_step("Map uncoded problems → ICD-9-CM CSV", map_cmd)

    dt = time.perf_counter() - t0
    log("=" * 60)
    status = "SUCCESS" if rc == 0 else "FAILED"
    log(f"ICD-9-CM PIPELINE {status}  in {dt:.1f}s")
    if Path(args.output).exists():
        log(f"Results CSV: {args.output}")
    log("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
