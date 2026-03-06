
# RAG + Tiny-LLM Edge Tests — ICD-10 Normalization

This repository is a minimal, local proof-of-concept that demonstrates a Retrieval-Augmented Generation (RAG) pipeline which: embed → retrieve → prompt → generate → validate. It converts free-text ED chief complaints into candidate ICD-10 codes using small, CPU-runnable models and a local LanceDB vector store.

Key goals:
- Fully local: no cloud APIs required for the core pipeline.
- CPU-friendly: supports tiny/quantized models for edge/CI use.
- Reproducible test-suite: nine curated edge cases exercise safety, parsing and grounding.

**Quick links**
- `01_build_fake_kb_lancedb.py` — build a synthetic LanceDB KB (ICD seed set + schema docs)
- `02_rag_llama32_edge_tests.py` — run the 9-case RAG + validation test harness

Repository layout
- `01_build_fake_kb_lancedb.py` — create and persist a small LanceDB vector store
- `02_rag_llama32_edge_tests.py` — orchestrates embed → retrieve → prompt → generate → validate
- `model_config.yaml` — model profile definitions and runtime flags
- `lancedb_store/` — local LanceDB files (ignored by `.gitignore`)

Requirements
- Python 3.10+ (3.12 tested in CI)
- Recommended packages (install with `pip`):

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` in this repo, install:

```bash
pip install lancedb sentence-transformers transformers torch python-dotenv pandas
```

Configuration notes
- The scripts default to a Windows-style LanceDB path used for development (`C:\AIgnatov\lancedb_store`). Edit the `db_dir` variable in both scripts to a Linux/macOS path when running locally.
- For gated models (e.g. Meta Llama 3.2), set `HF_TOKEN` in a `.env` file or export `HF_TOKEN` in your environment. Without a token the code falls back to an ungated tiny model.

Quickstart — build KB and run tests

```bash
# 1) (Optional) create a virtualenv
python -m venv .venv
source .venv/bin/activate

# 2) install deps
pip install lancedb sentence-transformers transformers torch python-dotenv pandas

# 3) build the fake knowledge base (run once)
python 01_build_fake_kb_lancedb.py

# 4) run the RAG edge tests
python -u 02_rag_llama32_edge_tests.py
```

Outputs
- Each test emits a validated JSON object that includes `candidate_icd_codes`, `confidence`, `flags` and `model_used`.
- Example result files are written to `/tmp` by the test harness (see script arguments).

Git / CI notes
- This repository ignores model caches, local LanceDB stores, env files and editor metadata via `.gitignore`.

Safety & grounding
- The prompt and post-processing enforce that suggested ICD codes must appear in the retrieved context — this reduces hallucination risk.
- The test-suite includes adversarial cases (empty input, prompt-injection requests, nonsense) to validate the pipeline's defensive behavior.

Next steps / extension ideas
- Replace the synthetic ICD seed with a full ICD-10 CSV import.
- Add a persistent results writer to SQL using `sql_fields_to_store`.
- Add CI job to run the edge-tests on PRs using a tiny local model to catch regressions.

If you'd like, I can now:
- Stage and commit these changes locally, and/or
- Add a remote named `local-clinical` and push (please provide the remote URL)

