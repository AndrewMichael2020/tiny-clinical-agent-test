# RAG + Tiny LLM Edge Tests — ICD-10 Clinical Code Normalization

A minimal proof-of-concept showing that a small LLM running entirely on CPU can normalize emergency-department chief complaints into candidate ICD-10 codes via Retrieval-Augmented Generation (RAG), with **no cloud API calls and no GPU required**.

---

## What this does

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_build_fake_kb_lancedb.py` | Embeds a synthetic knowledge base (hospital schema + ICD-10 codes) and writes it to a local [LanceDB](https://lancedb.github.io/lancedb/) vector store. |
| 2 | `02_rag_llama32_edge_tests.py` | Runs 9 edge-test cases through a full RAG pipeline: embed → retrieve → prompt → generate → validate structured JSON output. |

---

## Architecture

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
  Tiny LLM (Llama-3.2-1B  or  TinyLlama-1.1B-Chat, CPU)
       │
       ▼
  Structured JSON output ── validated ──► ICD candidates + confidence + flags
```

---

## Knowledge base (`01_build_fake_kb_lancedb.py`)

The KB contains two document types:

**Hospital schema** — three SQL `CREATE TABLE` stubs:
- `dbo.Patient` (MRN, DOB, Sex, PostalCode)
- `dbo.Encounter` (EncounterDtm, FacilityCode, ChiefComplaint, TriageAcuity)
- `dbo.Diagnosis` (DxCode, DxSystem, DxDescription, DxRank)

**ICD-10 codes** — ~600 codes spanning every major clinical system seen in the ED:

| System | Code range | Examples |
|--------|-----------|---------|
| Symptoms & Signs | R00–R99 | Chest pain, syncope, fever, haematuria, altered mental status |
| Cardiovascular | I10–I99 | Angina, STEMI, NSTEMI, AFib, PE, aortic dissection, DVT, heart failure |
| Respiratory | J00–J99 | URI, pharyngitis, pneumonia, COPD exacerbation, asthma, pneumothorax, ARDS |
| Gastrointestinal | K00–K95 | GERD, appendicitis, bowel obstruction, cholecystitis, pancreatitis, GI bleed |
| Genitourinary | N00–N99 | UTI, pyelonephritis, renal colic, kidney stones, PID, torsion |
| Neurology | G00–G99 | Meningitis, seizure, migraine, TIA, stroke, Bell palsy, Guillain-Barré |
| Mental Health | F00–F99 | Alcohol intoxication/withdrawal, opioid overdose, psychosis, depression, anxiety, panic, PTSD |
| Musculoskeletal | M00–M99 | Gout, OA, low back pain, sciatica, rotator cuff, fibromyalgia |
| Infectious | A00–B99 | Sepsis, C. diff, herpes zoster, HIV, hepatitis, Lyme |
| Endocrine/Metabolic | E00–E90 | DKA, hypoglycaemia, thyroid storm, electrolyte disorders, obesity |
| Skin | L00–L99 | Cellulitis, abscess, urticaria, Stevens-Johnson, drug eruption |
| Eye/Ear | H00–H95 | Conjunctivitis, acute glaucoma, otitis media, Ménière's, vertigo |
| Haematology | D50–D89 | Anaemia, DIC, ITP, neutropenia, sickle-cell crisis |
| Injury/Trauma | S00–T98 | Fractures, concussion, burns, overdose/poisoning, anaphylaxis |
| Pregnancy | O00–O9A | Spontaneous abortion, pre-eclampsia, hyperemesis, PPROM |
| Neoplasms | C00–D49 | Common ED cancer presentations |

All documents are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in LanceDB at `C:\AIgnatov\lancedb_store`.

---

## Test cases (`02_rag_llama32_edge_tests.py`)

Nine cases are run to cover the full edge-test matrix:

| Case ID | Triage | Purpose |
|---------|--------|---------|
| `GREEN_angina_like` | CTAS3 | Happy path — chest tightness maps to I20.9 / R07.9 |
| `GREEN_uri_like` | CTAS4 | Happy path — fever + sore throat maps to J06.9 |
| `GREEN_uti_like` | CTAS4 | Happy path — burning urination maps to N39.0 |
| `RED_empty` | CTAS5 | Empty complaint → must return `EMPTY_INPUT` flag, no ICD codes |
| `RED_nonsense` | CTAS5 | Garbage text → must return `NONSENSE_INPUT` flag, no ICD codes |
| `EDGE_contains_code` | CTAS4 | Complaint already mentions a code — test for hallucination / echo |
| `EDGE_conflicting_symptoms` | CTAS3 | Multi-system symptoms → must return `CONFLICTING_SYMPTOMS` flag |
| `EDGE_very_long` | CTAS3 | ~4 000-char complaint → truncation + token limit handling |
| `RED_schema_request` | NA | Prompt-injection-style input asking for schema fields, not clinical |

---

## Output JSON schema

Each case produces a validated JSON object:

```json
{
  "input_text": "<exact ChiefComplaint, char-for-char>",
  "normalized_chief_complaint": "<cleaned clinical text>",
  "candidate_icd_codes": ["I20.9"],
  "candidate_icd_rationales": ["Chest tightness on exertion, relieved by rest suggests angina."],
  "sql_fields_to_store": [
    "EncounterId", "ChiefComplaintRaw", "ChiefComplaintNormalized",
    "CandidateICD1", "CandidateICD1Confidence", "ModelName", "RunTimestampUTC"
  ],
  "confidence": 0.72,
  "flags": [],
  "model_used": "meta-llama/Llama-3.2-1B"
}
```

Validation enforces:
- All required keys are present
- `candidate_icd_codes` only contains codes present in the retrieved CONTEXT
- `input_text` matches the original complaint exactly
- `confidence` is in `[0, 1]`
- Placeholder schema echoes (e.g. `"string"`) are rejected

---

## Models used

| Role | Model | Notes |
|------|-------|-------|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` | 22 M params, CPU, ungated |
| Generator (preferred) | `meta-llama/Llama-3.2-1B` | Gated — requires HF token + Meta access |
| Generator (fallback) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Ungated — runs without a token |

The generator is loaded in `float16` with `low_cpu_mem_usage=True` to minimize RAM. Decoding is deterministic (greedy, no sampling).

---

## Setup

### Prerequisites
- Python 3.10+
- Windows path assumed for LanceDB store (`C:\AIgnatov\lancedb_store`) — edit the `db_dir` variable in both scripts if running on Linux/macOS.

### Install dependencies

```bash
pip install lancedb sentence-transformers transformers torch python-dotenv pandas
```

### Optional: Hugging Face token (for Llama-3.2-1B)

Create a `.env` file next to the scripts:

```
HF_TOKEN=hf_your_token_here
HF_HOME=C:\path\to\hf_cache
```

Without a token the pipeline automatically falls back to TinyLlama.

### Run

```bash
# Step 1: build the vector store (run once)
python 01_build_fake_kb_lancedb.py

# Step 2: run the edge tests
python -u 02_rag_llama32_edge_tests.py
```

---

## Key design decisions

- **CPU-only** — no GPU dependency; practical for local developer machines and Windows Server environments.
- **Grounded ICD codes** — the prompt strictly forbids codes not present in the retrieved context, reducing hallucination.
- **Retry with reinforcement** — on JSON parse/validate failure the pipeline appends stricter rules and retries once before raising.
- **Heartbeat threads** — long model-load and generation phases print periodic `HEARTBEAT` lines so the process never looks hung.
- **Behavior-only few-shot examples** — the in-prompt examples demonstrate empty/nonsense handling only, with no diagnosis-anchoring, to avoid biasing ICD predictions.

---

## Extending this

- **Expand the ICD seed set** — replace `build_fake_icd_docs()` with a full ICD-10 CSV import.
- **Add more tables** — extend `build_fake_schema_docs()` with your real EMR schema.
- **Swap the generator** — any `AutoModelForCausalLM`-compatible model works; update `candidates` in `load_generator_model()`.
- **Persist results** — write validated JSON objects to the `dbo.Diagnosis` table using the `sql_fields_to_store` list.
