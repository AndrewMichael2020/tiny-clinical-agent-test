# ED Chief Complaint → ICD-10 RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![CPU Only](https://img.shields.io/badge/Inference-CPU--only-orange?style=flat-square&logo=intel&logoColor=white)
![No GPU](https://img.shields.io/badge/GPU-not%20required-success?style=flat-square&logo=nvidia&logoColor=white)
![LanceDB](https://img.shields.io/badge/VectorDB-LanceDB-blueviolet?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗%20Models-Transformers-FFD21E?style=flat-square)
![RAG](https://img.shields.io/badge/Pattern-RAG-0078D4?style=flat-square)
![ICD-10](https://img.shields.io/badge/Coding-ICD--10-red?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQTEwIDEwIDAgMCAwIDIgMTJhMTAgMTAgMCAwIDAgMTAgMTAgMTAgMTAgMCAwIDAgMTAtMTBBMTAgMTAgMCAwIDAgMTIgMm0tMiAxNHYtNEg4bDQtNCA0IDRoLTJ2NHoiLz48L3N2Zz4=)
![Codespace](https://img.shields.io/badge/Dev-GitHub%20Codespace-181717?style=flat-square&logo=github&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow?style=flat-square)

A proof-of-concept demonstrating that a **sub-2 GB LLM running entirely on CPU** can normalize emergency-department (ED) chief complaints into candidate ICD-10 codes via Retrieval-Augmented Generation (RAG) — with no cloud API calls, no GPU, and no proprietary data.

> ⚠️ **Status:** Research / dev prototype. Not validated for clinical use.

---

## Overview

| Step | Script | What it does |
|------|--------|--------------|
| 1a | `01_build_fake_kb_lancedb.py` | Generates synthetic KB docs (hospital SQL schema + ~600 ICD-10 codes) and **writes them to `fake_kb_data.csv`**. |
| 1b | `01_build_fake_kb_lancedb.py` | Reads back the CSV, flushes the existing LanceDB table, embeds all docs, and re-ingests into [LanceDB](https://lancedb.github.io/lancedb/). |
| 2 | `02_rag_llama32_edge_tests.py` | Runs 9 structured edge-test cases through the full RAG pipeline: embed → retrieve → prompt → generate → parse → validate JSON. Auto-saves results to `results/rag_results_<timestamp>.json`. |
| — | `run_pipeline.py` | **One-command runner** that chains steps 1 and 2 end-to-end. |

---

## Architecture

```
Chief Complaint (free text)
        │
        ▼
  Sentence Embedder
  all-MiniLM-L6-v2  (22 M params, CPU)
        │
        ▼
  LanceDB cosine search ◄── KB: ~600 ICD-10 codes + 3 SQL schema docs
        │                         ▲
        │                         │  ingested from fake_kb_data.csv
        │                         │  (flushed + re-embedded on each build)
        │  top-k retrieved chunks
        ▼
  Prompt builder
  (structured system prompt + few-shot behavior examples)
        │
        ▼
  Tiny LLM  (1 B params, CPU-only, float16 or Q4_K_M)
        │
        ▼
  JSON output parser + validator
        │
        ▼
  { candidate_icd_codes, confidence, flags, … }
        │
        ▼
  results/rag_results_<timestamp>.json
```

---

## Model Profiles (`model_config.yaml`)

Four profiles are defined and benchmarked. Switch with `--model <profile>`, the `MODEL_PROFILE` env var, or by editing `active_profile` in `model_config.yaml`.

| Profile key | Model | Size | Dtype | Gated | Backend |
|---|---|---|---|---|---|
| `llama32_1b_instruct` *(default)* | `meta-llama/Llama-3.2-1B-Instruct` | ~2.5 GB | float16 | ✅ HF token + Meta licence | Transformers |
| `gemma3_1b` | `google/gemma-3-1b-it` | ~2.0 GB | bfloat16 | ✅ HF token + Google licence | Transformers |
| `danube3_500m` | `h2oai/h2o-danube3-500m-chat` | ~0.98 GB | float16 | ❌ ungated | Transformers |
| `llama32_1b_q4km` | `bartowski/Llama-3.2-1B-Instruct-GGUF` (Q4_K_M) | **~0.81 GB** | Q4_K_M | ❌ ungated | llama-cpp-python |

> **No HF token?** Use `danube3_500m` or `llama32_1b_q4km` — both run with zero credentials.

---

## Knowledge Base (`01_build_fake_kb_lancedb.py`)

The builder now follows a **CSV-first pipeline**:

1. **Generate** — `build_fake_schema_docs()` + `build_fake_icd_docs()` produce in-memory dicts.
2. **Persist to CSV** — `write_docs_to_csv()` serialises all docs to `fake_kb_data.csv` (columns: `doc_type`, `doc_id`, `title`, `text`, `icd_code`, `icd_desc`).
3. **Reload from CSV** — `read_docs_from_csv()` reads the CSV back, making the CSV the authoritative source for downstream ingestion.
4. **Flush** — `flush_lancedb_table()` drops the existing `kb_docs` table for a clean slate.
5. **Embed + ingest** — Docs are vectorised with `all-MiniLM-L6-v2` and written to `lancedb_store/`.

> Override paths via env vars: `KB_CSV_PATH` (CSV output), `LANCEDB_DIR` (store directory).

Two document types are embedded and stored in `./lancedb_store`:

### Hospital schema stubs
Three SQL `CREATE TABLE` definitions:
- `dbo.Patient` — MRN, DOB, Sex, PostalCode
- `dbo.Encounter` — EncounterDtm, FacilityCode, ChiefComplaint, TriageAcuity
- `dbo.Diagnosis` — DxCode, DxSystem, DxDescription, DxRank

### ICD-10 code index (~600 codes)

| Clinical domain | Code range | Sample conditions |
|---|---|---|
| Symptoms & Signs | R00–R99 | Chest pain, syncope, fever, haematuria, altered mental status |
| Cardiovascular | I10–I99 | Angina, STEMI, NSTEMI, AFib, PE, aortic dissection, DVT, heart failure |
| Respiratory | J00–J99 | URI, pharyngitis, pneumonia, COPD exacerbation, asthma, pneumothorax |
| Gastrointestinal | K00–K95 | GERD, appendicitis, bowel obstruction, cholecystitis, pancreatitis, GI bleed |
| Genitourinary | N00–N99 | UTI, pyelonephritis, renal colic, kidney stones, PID, torsion |
| Neurology | G00–G99 | Meningitis, seizure, migraine, TIA, stroke, Bell palsy |
| Mental Health / Tox | F00–F99 | Alcohol withdrawal, opioid OD, psychosis, depression, panic, PTSD |
| Musculoskeletal | M00–M99 | Gout, OA, low back pain, sciatica, rotator cuff, fibromyalgia |
| Infectious Disease | A00–B99 | Sepsis, C. diff, herpes zoster, HIV, hepatitis, Lyme |
| Endocrine / Metabolic | E00–E90 | DKA, hypoglycaemia, thyroid storm, electrolyte disorders |
| Dermatology | L00–L99 | Cellulitis, abscess, urticaria, Stevens-Johnson |
| Eye / ENT | H00–H95 | Conjunctivitis, acute glaucoma, otitis media, Ménière's, vertigo |
| Haematology | D50–D89 | Anaemia, DIC, ITP, neutropenia, sickle-cell crisis |
| Injury / Trauma | S00–T98 | Fractures, concussion, burns, poisoning, anaphylaxis |
| Obstetrics | O00–O9A | Spontaneous abortion, pre-eclampsia, hyperemesis, PPROM |
| Neoplasms | C00–D49 | Common oncology presentations seen in the ED |

---

## Test Suite (`02_rag_llama32_edge_tests.py`)

Nine cases cover three behavioural categories:

| Case ID | Category | CTAS | What is being tested |
|---|---|---|---|
| `GREEN_angina_like` | GREEN | 3 | Happy path — chest tightness → I20.9 / R07.9 |
| `GREEN_uri_like` | GREEN | 4 | Happy path — fever + sore throat → J06.9 |
| `GREEN_uti_like` | GREEN | 4 | Happy path — pelvic pain + urgency → N39.0 |
| `RED_empty` | RED | 5 | Empty input → `EMPTY_INPUT` flag, zero ICD codes |
| `RED_nonsense` | RED | 5 | Garbage text → `NONSENSE_INPUT` / `LOW_CONTEXT` flag, zero ICD codes |
| `RED_schema_request` | RED | — | Prompt-injection asking for DB schema → rejected, no ICD codes |
| `EDGE_contains_code` | EDGE | 4 | Complaint already contains an ICD code string → test for echo / hallucination |
| `EDGE_conflicting_symptoms` | EDGE | 3 | Multi-system symptoms → `CONFLICTING_SYMPTOMS` flag |
| `EDGE_very_long` | EDGE | 3 | ~1 800-token input → truncation + token limit handling |

---

## Output Schema

Every case produces a validated JSON object:

```json
{
  "input_text": "<exact ChiefComplaint text, char-for-char>",
  "normalized_chief_complaint": "<cleaned clinical phrase>",
  "candidate_icd_codes": ["I20.9"],
  "candidate_icd_rationales": ["Chest tightness on exertion, relieved at rest — consistent with stable angina."],
  "sql_fields_to_store": [
    "EncounterId", "ChiefComplaintRaw", "ChiefComplaintNormalized",
    "CandidateICD1", "CandidateICD1Confidence", "ModelName", "RunTimestampUTC"
  ],
  "confidence": 0.72,
  "flags": [],
  "model_used": "meta-llama/Llama-3.2-1B-Instruct"
}
```

**Validation rules enforced on every output:**
- All required keys present
- `candidate_icd_codes` contains only codes that appear in the retrieved context (grounding check)
- `input_text` matches the original complaint exactly
- `confidence` in `[0.0, 1.0]`
- Placeholder strings (e.g. `"string"`) are rejected
- On parse or validation failure: prompt is reinforced and retried once before raising

---

## Benchmark Results Summary

Tested on GitHub Codespace — AMD EPYC 9V74, 4 vCPUs, 32 GB RAM, no GPU.

| Model | Pass rate (9 cases) | Avg tokens/sec | Peak RAM |
|---|---|---|---|
| Llama 3.2 1B Instruct (float16) | see `MODEL_COMPARISON_REPORT.md` | ~8–12 tok/s | ~2.5 GB |
| Gemma 3 1B Instruct (bfloat16) | see report | ~8–11 tok/s | ~2.0 GB |
| H2O-Danube3 500M (float16) | see report | ~14–18 tok/s | ~0.98 GB |
| Llama 3.2 1B Q4_K_M (GGUF) | see report | ~6–9 tok/s | ~0.81 GB |

Full per-case pass/fail table and timing data: [`MODEL_COMPARISON_REPORT.md`](MODEL_COMPARISON_REPORT.md)

---

## Setup

### Prerequisites

- Python 3.10+
- ~3 GB free disk for model weights (less for GGUF / Danube3)
- No GPU required

### 1 — Clone and install dependencies

```bash
pip install lancedb sentence-transformers transformers torch python-dotenv pyyaml pandas
```

For the GGUF profile, also install:

```bash
pip install llama-cpp-python
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN if using a gated model, adjust HF_HOME to your cache path
```

### 3 — Build the vector store

```bash
python 01_build_fake_kb_lancedb.py
```

This runs the full CSV-first pipeline:
- Generates 605 KB docs (3 schema + 602 ICD-10)
- Writes them to `fake_kb_data.csv`
- Flushes any existing `kb_docs` table from LanceDB
- Embeds all docs and re-ingests into `./lancedb_store/`

### 4 — Run the edge tests

```bash
# Default profile (llama32_1b_instruct) — saves JSON automatically
python -u 02_rag_llama32_edge_tests.py

# Ungated alternative (no HF token needed)
python -u 02_rag_llama32_edge_tests.py --model danube3_500m

# Specify custom results path
python -u 02_rag_llama32_edge_tests.py --save-results my_results.json

# Via environment variable
MODEL_PROFILE=gemma3_1b python -u 02_rag_llama32_edge_tests.py
```

Results are always persisted to `results/rag_results_<YYYYMMDD_HHMMSS>.json` (unless overridden with `--save-results`).

### 5 — Run the full pipeline in one command

```bash
# Uses active_profile from model_config.yaml
python run_pipeline.py

# Override model
python run_pipeline.py --model danube3_500m

# Override output path
python run_pipeline.py --save-results path/to/output.json
```

`run_pipeline.py` chains steps 3 and 4: build KB from CSV → ingest LanceDB → run RAG tests → save JSON.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **CPU-only inference** | Targets Windows Server / developer laptops with no GPU; PyTorch CPU threads saturate available cores. |
| **Grounded ICD codes only** | The prompt forbids codes absent from the retrieved context, directly reducing hallucination. |
| **Retry with reinforcement** | On JSON parse / validation failure, stricter rules are appended and generation is retried once. |
| **Heartbeat threads** | Long model-load and generation phases emit periodic `HEARTBEAT` lines so the process never looks hung in CI or terminal. |
| **Behaviour-only few-shot examples** | In-prompt examples demonstrate empty/nonsense handling only — no diagnosis-anchoring — to avoid biasing ICD predictions. |
| **`model_config.yaml` profiles** | All model hyperparameters (dtype, max_new_tokens, repetition_penalty, backend) live in a single YAML file; switching models requires no code changes. |

---

## Extending This

- **Swap in real data** — Replace `build_fake_icd_docs()` with a reader for a real ICD-10 CSV (CMS tabular file or WHO release); it will be automatically written to `fake_kb_data.csv` and ingested.
- **Add real EMR schema** — Extend `build_fake_schema_docs()` with your actual `CREATE TABLE` definitions.
- **Swap the generator** — Any `AutoModelForCausalLM`-compatible model works; add a new profile block to `model_config.yaml`.
- **Persist results** — The `results/rag_results_<timestamp>.json` file includes full `case_outputs` per case; feed those into your pipeline or BI tool.
- **Batch mode** — Wrap `run_case()` in a loop over a CSV of real chief complaints, or extend `build_test_cases()` to load from a CSV.

---

## File Reference

```
.
├── 01_build_fake_kb_lancedb.py   # KB builder: generate → CSV → flush → embed → LanceDB
├── 02_rag_llama32_edge_tests.py  # RAG pipeline + 9-case edge-test harness (auto-saves JSON)
├── run_pipeline.py               # One-command runner: KB build → RAG tests → JSON results
├── fake_kb_data.csv              # Generated KB docs (CSV intermediary; auto-created by step 1)
├── model_config.yaml             # Model profiles (switch without code changes)
├── lancedb_store/                # Persisted vector store (flushed + rebuilt by step 1)
├── results/                      # Per-run JSON results (created by step 2 / run_pipeline.py)
├── MODEL_COMPARISON_REPORT.md    # Benchmark results across all four model profiles
├── TEST_REPORT.md                # Detailed per-case pass/fail output
├── .env.example                  # Environment variable template (copy → .env)
└── .gitignore
```
