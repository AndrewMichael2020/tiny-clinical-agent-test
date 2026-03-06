
# RAG + Tiny LLM Edge Tests — ICD-10 Clinical Code Normalization

**Generated:** 2026-03-05

This repository contains a minimal proof-of-concept showing how a small LLM running entirely on CPU can normalize emergency-department chief complaints into candidate ICD-10 codes using a Retrieval-Augmented Generation (RAG) pipeline. The focus is on reproducible, local execution (no cloud API calls required) and defensive validation to reduce hallucinations.

---

## 1. What this does

The codebase implements a simple pipeline:

- Embed the incoming chief complaint with `sentence-transformers/all-MiniLM-L6-v2` (CPU).
- Retrieve relevant ICD-10 and schema docs from a local LanceDB vector store.
- Build a structured few-shot prompt containing grounding context and behavior examples.
- Generate structured JSON output from a small generator model (HF or llama-cpp-backed GGUF).
- Post-process and validate the JSON against the retrieved context, applying confidence floors and normalization fixes.

Two scripts are provided:

- `01_build_fake_kb_lancedb.py`: build a synthetic LanceDB KB (hospital schema stubs + ~600 ICD-10 seed docs).
- `02_rag_llama32_edge_tests.py`: run a 9-case edge test suite exercising happy-path, adversarial and boundary behaviors.

---

## 2. System & Test Context

See [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) for full run metadata (hardware, timings, models tested, and behavior notes). This work was executed on Ubuntu 24.04.3 LTS with CPU-only inference.

---

## 3. Architecture

ASCII overview:

```
Chief Complaint (text)
	  │
	  ▼
  Embedder (all-MiniLM-L6-v2, CPU)
	  │
	  ▼
  LanceDB vector search  ←── KB: ICD-10 codes + DB schema docs
	  │
	  ▼
  Prompt builder (structured, few-shot behavior examples)
	  │
	  ▼
  Generator (HF / gguf via llama-cpp-python)
	  │
	  ▼
  Structured JSON output ── validated ──► ICD candidates + confidence + flags
```

Key components:

- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (CPU-only, small, fast).
- Vector DB: LanceDB storing embedded ICD docs and lightweight schema examples.
- Generator: flexible backend — HuggingFace Transformers (fp16) or `llama-cpp-python` (GGUF Q4_K_M) for quantized local runs.
- Validation: ensures required keys, confidence range, and that output ICDs appear in retrieved context.

---

## 4. Knowledge base (details)

`01_build_fake_kb_lancedb.py` creates two document families:

- Hospital schema stubs (`dbo.Patient`, `dbo.Encounter`, `dbo.Diagnosis`).
- ICD-10 seed documents (~600 codes) spanning major clinical systems (R00–R99, I10–I99, J00–J99, K00–K95, N00–N99, etc.).

Embeddings are generated with the MiniLM model and written to `lancedb_store/` by default (edit `db_dir` in the scripts to change the path).

---

## 5. Test suite (9 cases)

The test harness runs these curated cases to validate behavior:

- 3 GREEN (happy path clinical inputs)
- 3 RED (adversarial / invalid inputs: empty, nonsense, prompt-injection)
- 3 EDGE (boundary conditions: already-containing-code, conflicting symptoms, very long input)

Each case produces a validated JSON object with keys such as `input_text`, `normalized_chief_complaint`, `candidate_icd_codes`, `candidate_icd_rationales`, `confidence`, `flags`, and `model_used`.

Validation rules include:

- All required keys present.
- `candidate_icd_codes` must be drawn from retrieved context.
- `confidence` must be in `[0, 1]` (post-processing may floor low confidences when codes are grounded).
- Placeholder or schema-echo responses are rejected and trigger a retry path.

---

## 6. Models and behavior notes

Refer to [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) for complete comparative results. Highlights:

- Llama 3.2 1B (fp16) — clean JSON, good calibration.
- Gemma 3 1B — noisier JSON, required bracket-matched extraction and confidence-floor fixes.
- H2O Danube3 500M — smallest weights, fast to load, required `no_system_role` workaround for chat template.
- Llama Q4_K_M (GGUF via llama-cpp-python) — fastest CPU throughput, smallest disk footprint, excellent JSON quality.

---

## 7. Setup & Quickstart

Prereqs: Python 3.10+ (3.12 used during testing)

Install dependencies:

```bash
pip install lancedb sentence-transformers transformers torch python-dotenv pandas
```

Optional: create a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

Build the fake KB (once):

```bash
python 01_build_fake_kb_lancedb.py
```

Run the edge tests:

```bash
python -u 02_rag_llama32_edge_tests.py
```

Notes:

- The scripts default to a Windows-style LanceDB path used by the project author. Change `db_dir` inside the scripts to point to a Linux/macOS path when running here.
- For gated HF models, set `HF_TOKEN` in a `.env` file or as an environment variable; otherwise the code falls back to an ungated tiny model.

---

## 8. Safety, grounding and pipeline fixes

Key robustness measures applied in the pipeline:

- Bracket-matched `extract_json()` with fallbacks (first-balanced-braces, fence stripping).
- Confidence floor applied in `post_process()` when grounded ICD candidates exist but `confidence` is reported as `0`.
- Coercion of list-valued `normalized_chief_complaint` to a joined string when models return a list.
- Prompt-injection detection: inputs that explicitly request schema or data are rejected with a `RED_schema_request` result and not sent to the LLM.

---

## 9. Extending this project

- Replace `build_fake_icd_docs()` with an ICD-10 CSV import to expand coverage.
- Persist validated outputs to a SQL store using the `sql_fields_to_store` output field.
- Add a CI job that runs the lightweight edge-tests on PRs using the quantized GGUF model to catch regressions early.

---

## 10. Where to go next (I can help)

- Restore or expand the ICD seed set from an authoritative CSV.
- Add persistence (SQL writer) for validated outputs.
- Create a `requirements.txt` and a small GitHub Actions job to run the edge tests on PRs.

If you want, I will now stage, commit and (optionally) add the `local-clinical` remote and push — provide the remote URL or tell me to prompt for it.

