# 03_build_icd9_kb.py — Build ICD-9-CM knowledge base in LanceDB
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import csv
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import lancedb
from sentence_transformers import SentenceTransformer


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Header-row detection ──────────────────────────────────────────────────────
# icd_9_dictionary.csv contains 19 synthetic section-header rows whose
# DiagnosisCode is R01–R17, "V" (single letter), or "UNK".
# These are NOT valid ICD-9-CM codes and must be excluded.
# V-codes with digits (e.g. V01, V01.1) ARE valid supplementary ICD-9-CM codes.
_HEADER_RE = re.compile(r'^R\d+$')

def _is_header_row(code: str) -> bool:
    return bool(_HEADER_RE.match(code)) or code in ("V", "UNK")


def load_icd9_codes(csv_path: Path) -> List[Dict[str, str]]:
    """Read icd_9_dictionary.csv, skip header rows, deduplicate by code."""
    seen: set[str] = set()
    codes: List[Dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("DiagnosisCode") or "").strip()
            desc = (row.get("DiagnosisDescr") or "").strip()
            if not code or not desc:
                continue
            if _is_header_row(code):
                continue
            if code in seen:
                continue          # keep first occurrence per code
            seen.add(code)
            codes.append({"code": code, "desc": desc})

    return codes


def build_icd9_docs(codes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert raw ICD-9 code dicts to the standard KB document format."""
    docs: List[Dict[str, Any]] = []
    for entry in codes:
        code = entry["code"]
        desc = entry["desc"]
        docs.append({
            "doc_type": "icd",
            "doc_id":   f"icd:{code}",
            "title":    f"ICD9 {code}",
            "text":     f"{code} | {desc}",
            "icd_code": code,
            "icd_desc": desc,
        })
    return docs


def flush_table(db_dir: Path, table_name: str) -> None:
    """Drop a LanceDB table if it exists."""
    db = lancedb.connect(str(db_dir))
    existing = db.list_tables()
    if table_name in existing:
        db.drop_table(table_name)
        log(f"Flushed existing LanceDB table: '{table_name}'")
    else:
        log(f"Table '{table_name}' did not exist — nothing to flush")


def main() -> int:
    t0 = time.perf_counter()
    log("START 03_build_icd9_kb.py")
    log(f"Python exe: {sys.executable}")
    log(f"CWD: {os.getcwd()}")

    script_dir = Path(__file__).resolve().parent
    db_dir     = Path(os.environ.get("LANCEDB_DIR", str(script_dir / "lancedb_store")))
    db_dir.mkdir(parents=True, exist_ok=True)

    icd9_table = "kb_docs_icd9"
    old_table  = "kb_docs"          # stale ICD-10 table from previous runs

    # ── Step 1: Load ICD-9-CM dictionary ─────────────────────────────────────
    dict_path = Path(os.environ.get("ICD9_DICT_PATH", str(script_dir / "icd_9_dictionary.csv")))
    log(f"STEP 1 — Loading ICD-9-CM dictionary: {dict_path}")
    codes = load_icd9_codes(dict_path)
    log(f"  Loaded {len(codes)} unique ICD-9-CM codes (after filtering headers + dedup)")

    docs = build_icd9_docs(codes)
    log(f"  Built {len(docs)} KB documents")

    # ── Step 2: Flush LanceDB tables ─────────────────────────────────────────
    log("STEP 2 — Flushing LanceDB tables")
    flush_table(db_dir, icd9_table)   # flush ICD-9 table (clean slate)
    flush_table(db_dir, old_table)    # flush stale ICD-10 table

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    # Load embedder name from model_config.yaml so KB and mapper always agree.
    _cfg_path = Path(__file__).resolve().parent / "model_config.yaml"
    try:
        import yaml
        with open(_cfg_path) as _f:
            _cfg = yaml.safe_load(_f)
        embedder_name = _cfg.get("embedder", "NeuML/pubmedbert-base-embeddings")
    except Exception:
        embedder_name = "NeuML/pubmedbert-base-embeddings"
    log(f"STEP 3 — Embedding {len(docs)} docs with {embedder_name}")
    t_emb = time.perf_counter()
    embedder = SentenceTransformer(embedder_name, device="cpu")
    log(f"  Embedder ready in {time.perf_counter() - t_emb:.2f}s")

    texts = [d["text"] for d in docs]
    t_vec = time.perf_counter()
    vecs = embedder.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    log(f"  Embedding done in {time.perf_counter() - t_vec:.2f}s  shape={vecs.shape}")

    for i in range(len(docs)):
        docs[i]["vector"] = vecs[i].tolist()

    # ── Step 4: Write to LanceDB ──────────────────────────────────────────────
    log(f"STEP 4 — Writing {len(docs)} docs to LanceDB table '{icd9_table}'")
    db = lancedb.connect(str(db_dir))
    db.create_table(icd9_table, data=docs, mode="overwrite")

    table = db.open_table(icd9_table)
    import pandas as pd
    preview = table.to_pandas().head(5)
    log("Preview (first 5 rows):")
    print(preview[["doc_type", "doc_id", "title", "text"]].to_string(index=False), flush=True)

    log(f"DONE in {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("ERROR")
        traceback.print_exc()
        raise SystemExit(1)
