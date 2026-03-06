#!/usr/bin/env python3
"""run_pipeline.py — End-to-end pipeline runner.

Steps executed in order:
  1. Generate fake KB docs → write to CSV  (01_build_fake_kb_lancedb.py)
  2. Ingest CSV → embed → store in LanceDB  (01_build_fake_kb_lancedb.py, cont.)
  3. Run RAG edge tests with the active model profile  (02_rag_llama32_edge_tests.py)
  4. Persist per-case JSON results to ./results/

Usage:
    python run_pipeline.py                        # uses active_profile from model_config.yaml
    python run_pipeline.py --model danube3_500m   # override model profile
    python run_pipeline.py --save-results my.json # custom output path
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
    _default_results = (
        Path(__file__).resolve().parent / "results"
        / f"rag_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    parser = argparse.ArgumentParser(
        description="Full pipeline: generate CSV → ingest LanceDB → run RAG tests → save JSON"
    )
    parser.add_argument(
        "--model", metavar="PROFILE",
        help="Model profile from model_config.yaml (overrides active_profile)",
    )
    parser.add_argument(
        "--save-results", metavar="PATH",
        default=str(_default_results),
        help=f"JSON output path (default: {_default_results})",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    log("=" * 60)
    log("PIPELINE START")
    log("=" * 60)

    python = sys.executable
    script_dir = Path(__file__).resolve().parent

    # Step 1+2 — Build fake KB: generate CSV then ingest into LanceDB
    rc = run_step(
        "Build fake KB: generate CSV + embed + ingest LanceDB",
        [python, str(script_dir / "01_build_fake_kb_lancedb.py")],
    )
    if rc != 0:
        log("Aborting pipeline — KB build failed.")
        return rc

    # Step 3 — Run RAG edge tests + save JSON
    rag_cmd = [python, str(script_dir / "02_rag_llama32_edge_tests.py"),
               "--save-results", args.save_results]
    if args.model:
        rag_cmd += ["--model", args.model]

    rc = run_step("RAG edge tests + JSON output", rag_cmd)

    dt = time.perf_counter() - t0
    log("=" * 60)
    status = "SUCCESS" if rc == 0 else "FAILED"
    log(f"PIPELINE {status}  in {dt:.1f}s")
    if Path(args.save_results).exists():
        log(f"Results JSON: {args.save_results}")
    log("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
