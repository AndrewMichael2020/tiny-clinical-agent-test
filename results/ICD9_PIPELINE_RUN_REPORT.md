# ICD-9-CM Auto-Coding Pipeline — Final Run Report

**Date:** 2026-03-07  
**Run:** Definitive (Run 3 of 3)  
**Output:** `icd9_mapping_results.csv` (workspace root)

---

## 1. Pipeline Summary

| Item | Value |
|---|---|
| Input | `mock_uncoded.csv` — 100 patient problem notes |
| Knowledge Base | `icd_9_dictionary.csv` — 7,650 unique ICD-9-CM codes (LanceDB `kb_docs_icd9`) |
| Embedder | `all-MiniLM-L6-v2` (sentence-transformers) |
| LLM | Llama 3.2 1B Instruct Q4_K_M (GGUF, CPU) |
| Retrieval | Top-8 ANN L2 search |
| Runtime | **852.4 s (~14.2 min)** |
| Cases processed | **100** |
| Cases with codes | **93** (7 bypassed deterministically) |
| Mean overall confidence | **0.890** |

---

## 2. Output Schema (12 columns)

| Column | Description |
|---|---|
| `SyntheticProblemKey` | Problem identifier |
| `SyntheticPatientKey` | Patient identifier |
| `PatientProblemDescription` | Raw input text |
| `NormalizedDescription` | Concise clinical phrase produced by LLM (2–5 words) |
| `CandidateICD9Codes` | Pipe-separated ICD-9-CM codes (1–3) |
| `CandidateICD9Descs` | Corresponding descriptions |
| `CandidateICD9ModelConf` | **NEW** — Model's per-code confidence (pipe-separated floats; blank if not self-reported) |
| `CandidateICD9Similarity` | **NEW** — Retrieval similarity per code (1 − dist/√2, always computed from LanceDB) |
| `Rationale` | Model's reasoning per code |
| `Confidence` | Overall scalar confidence (0–1) |
| `Flags` | Quality flags (see §4) |
| `ModelUsed` | `Llama 3.2 1B Q4_K_M (GGUF)` |

---

## 3. Code Assignment Distribution

| Codes assigned | Cases |
|---|---|
| 1 code | 6 |
| 2 codes | 24 |
| 3 codes | 63 |
| 0 (deterministic bypass) | 7 |

**Most cases (87%) received 2–3 candidate codes**, giving clinicians meaningful options while staying within the 1–3 cap imposed by the prompt.

---

## 4. Quality Flags

| Flag | Count | Notes |
|---|---|---|
| `NONSENSE_INPUT` | 5 | Admin/behavioural bypasses (see §5) |
| `LOW_CONTEXT` | 1 | Retrieval distance too far (> 1.10 threshold) |
| None | 94 | Processed normally |

---

## 5. Deterministic Bypass Cases (no LLM call)

| Input | Reason |
|---|---|
| `UNABLE TO CONTACT PT` | Admin exact-match |
| `UNABLE TO CONTACT PATIENT` | Admin exact-match |
| `Care Team:` | Trailing-colon admin label |
| `SVT` | ⚠️ False positive — too-short abbreviation (3 chars); SVT = Supraventricular Tachycardia is valid. Known limitation. |

---

## 6. Known Edge Cases

### "Treated for TB" — empty after procedure-code stripping
The model returned only ICD-9 Vol. 3 procedure codes (`18.01`, `17.45`, `11.94`) for this input.
The procedure-code stripper correctly removed them (no procedure keyword in input),
leaving an empty code list with no flag set. Root cause: TB diagnosis codes (010–018.x) 
were not retrieved/selected despite being in the KB. Likely a short-text retrieval gap.

### "SVT" — false positive NONSENSE_INPUT  
The 3-character abbreviation trips the short-input nonsense gate. 
A future fix would add a clinical-acronym whitelist (SVT, DVT, UTI, CHF, GERD, COPD, etc.).

### Gallbladder polyp — KB gap  
ICD-9 dictionary has no specific "polyp of gallbladder" entry; closest codes are `575.6`
(Cholesterolosis) and `575.8`. Retrieved context skews to gallstone codes; result is 
defensible but not ideal. This is a dictionary limitation, not a pipeline bug.

---

## 7. Per-Code Scoring (New in Run 3)

Two independent confidence signals now appear for every code:

**`CandidateICD9ModelConf`** — The model's own self-reported confidence per candidate code.  
Expressed as a pipe-separated list matching `CandidateICD9Codes`. Blank entries mean the 
model did not volunteer a score for that code (not fabricated).

**`CandidateICD9Similarity`** — Retrieval similarity computed from LanceDB L2 distance.  
Formula: `max(0.0, 1.0 − distance / √2)`. Always computed; empty only if the code was 
not among the retrieved candidates (e.g., fallback codes).

| Metric | Mean | Min | Max |
|---|---|---|---|
| Model confidence (per code) | 0.827 | 0.600 | 1.000 |
| Retrieval similarity (per code) | 0.439 | 0.139 | 0.836 |

**Similarity bands across all assigned codes:**
- High (> 0.60): 36 code assignments — strong KB match
- Medium (0.40–0.60): 106 code assignments — reasonable match
- Low (< 0.40): 101 code assignments — weaker retrieval; code chosen by model reasoning

The gap between model confidence (mean 0.83) and retrieval similarity (mean 0.44) reflects 
the 1B model's tendency to inflate self-reported confidence. Retrieval similarity is the 
more objective signal.

---

## 8. Prompt Engineering Iterations (Runs 1 → 3)

| Problem (Run 1) | Fix applied |
|---|---|
| Model wrote narrative sentences ("The patient has…") | Changed normalisation target: "concise clinical phrase (2–5 words)" |
| Model picked up to 8 codes including procedure codes | Capped to 1–3; added procedure preference; post-process procedure stripper |
| "Care Team:" → 8 hallucinated codes | Trailing-colon admin bypass |
| Model put ICD descriptions in `flags` field | `_VALID_FLAGS` whitelist; non-valid strings silently dropped |
| NONSENSE_INPUT over-fired on valid clinical text | Removed instruction from prompt; handled deterministically only |
| Low retrieval quality not signalled | Retrieval distance gate (threshold 1.10) → `LOW_CONTEXT` |

---

## 9. Architecture

```
PatientProblemDescription (raw text)
        │
        ▼
┌────────────────────────┐
│ detect_complaint_type()│  → admin / nonsense
│  · exact match         │       │
│  · trailing-colon RE   │       ▼  deterministic bypass
│  · behavioural RE      │  build_deterministic_result()
└────────────────────────┘       (no codes, NONSENSE_INPUT flag)
        │ normal
        ▼
┌────────────────────────┐
│ all-MiniLM-L6-v2       │  embed query
│ LanceDB ANN search     │  top-8, L2
└────────────────────────┘
        │ retrieved_rows + distances
        │ (gate: min_distance < 1.10)
        ▼
┌────────────────────────┐
│ Llama 3.2 1B Q4_K_M    │  chat or plain prompt
│ (llama-cpp-python)     │  → JSON {codes, descs, confs, rationale, flags}
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ post_process()         │  normalize_keys / normalize_flags
│  · grounding check     │  strip procedure codes
│  · procedure stripper  │  compute CandidateICD9Similarity
│  · per-code scoring    │  validate flags whitelist
└────────────────────────┘
        │
        ▼
icd9_mapping_results.csv  (100 rows × 12 columns)
```

---

## 10. Files Produced / Modified

| File | Role |
|---|---|
| `icd9_mapping_results.csv` | Final output — 100 rows, 12 columns |
| `03_build_icd9_kb.py` | Builds LanceDB KB (`kb_docs_icd9`, 7,650 docs) |
| `04_map_uncoded_problems.py` | Main RAG mapper (all prompt + scoring logic) |
| `run_icd9_pipeline.py` | Orchestrator (step 03 → step 04) |
| `requirements.txt` | Pinned dependencies |
| `model_config.yaml` | Active profile: `llama32_1b_q4km` |
| `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Downloaded GGUF model (808 MB) |
| `hf_cache/` | HuggingFace cache (embedder weights) |
