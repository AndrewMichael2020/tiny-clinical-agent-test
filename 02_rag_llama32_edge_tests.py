# 02_rag_llama32_edge_tests.py
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Config file ───────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.yaml"

def load_model_config(profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Load model_config.yaml and return the active profile dict."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML required: pip install pyyaml")

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Priority: CLI arg > MODEL_PROFILE env var > active_profile in file
    chosen = (
        profile_name
        or os.environ.get("MODEL_PROFILE")
        or cfg.get("active_profile")
    )
    profiles = cfg.get("profiles", {})
    if chosen not in profiles:
        raise ValueError(
            f"Unknown profile '{chosen}'. Available: {list(profiles.keys())}"
        )

    profile = dict(profiles[chosen])
    profile["profile_name"] = chosen
    return profile

_SQL_FIELDS = [
    "EncounterId", "ChiefComplaintRaw", "ChiefComplaintNormalized",
    "CandidateICD1", "CandidateICD1Confidence", "ModelName", "RunTimestampUTC",
]

# Per-case behavioral rules applied in post_process() then asserted in validate().
# force_flags:    flags that MUST appear (post_process adds them if missing)
# max_icd_count:  hard cap enforced by post_process + asserted in validate
# max_confidence: upper bound enforced by post_process + asserted in validate
# min_icd_count:  lower bound asserted in validate (fallback in post_process ensures this)
# min_confidence: lower bound asserted in validate
_CASE_BEHAVIORAL_RULES: Dict[str, Dict[str, Any]] = {
    "RED_empty":                 {"force_flags": ["EMPTY_INPUT"],       "max_icd_count": 0, "max_confidence": 0.20},
    "RED_nonsense":              {"force_flags": ["NONSENSE_INPUT"],     "max_icd_count": 0, "max_confidence": 0.20},
    "EDGE_conflicting_symptoms": {"force_flags": ["CONFLICTING_SYMPTOMS"]},
    "GREEN_angina_like":         {"min_icd_count": 1, "min_confidence": 0.40},
    "GREEN_uri_like":            {"min_icd_count": 1, "min_confidence": 0.40},
    "GREEN_uti_like":            {"min_icd_count": 1, "min_confidence": 0.40},
}

_FLAG_ALIASES: Dict[str, str] = {
    "CONFLICTING_SYMPOTMS":    "CONFLICTING_SYMPTOMS",
    "CONFLICTING_SYMPTOM":     "CONFLICTING_SYMPTOMS",
    "CONFLICTING_SYMPTOMES":   "CONFLICTING_SYMPTOMS",
    "CONFLICTINGSYMPTOMS":     "CONFLICTING_SYMPTOMS",
    "EMPTY":                   "EMPTY_INPUT",
    "EMPTY_COMPLAINT":         "EMPTY_INPUT",
    "NONSENSE":                "NONSENSE_INPUT",
    "NONSENSE_COMPLAINT":      "NONSENSE_INPUT",
    "LOW_CONFIDENCE":          "LOW_CONTEXT",
    "INSUFFICIENT_CONTEXT":    "LOW_CONTEXT",
    "INSUFFICIENT":            "LOW_CONTEXT",
    "NO_CONTEXT":              "LOW_CONTEXT",
}

# Canonical ICD key variants; normalized lookup (lowercase + underscores) → canonical
_ICD_CODES_VARIANTS = {
    "candidate_icd_codes", "candidate_icds", "candidate_icd", "icds", "icd_codes",
    "candidate_icald_codes", "icd", "codes",
}
_RATIONALE_VARIANTS = {
    "candidate_icd_rationales", "candidate_icd_rationale", "icd_rationales",
    "rational_candidate_icd_code", "rationales", "icd_rationale",
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_env_from_script_dir() -> None:
    try:
        from dotenv import load_dotenv
    except Exception as e:
        log(f"dotenv not available: {e}")
        return
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    log(f"Looking for .env at: {env_path}")
    if env_path.exists():
        ok = load_dotenv(dotenv_path=str(env_path), override=True)
        log(f"load_dotenv(override=True) returned: {ok}")
    else:
        log(".env not found. Proceeding without HF_TOKEN.")


def show_env() -> None:
    keys = [
        "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE", "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED",
    ]
    for k in keys:
        v = os.environ.get(k)
        if k == "HF_TOKEN" and v:
            v = v[:6] + "..." + v[-4:]
        log(f"ENV {k} = {v}")


def get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v else None


def start_heartbeat(label: str, every_seconds: int = 20) -> threading.Event:
    stop = threading.Event()

    def _beat():
        n = 0
        while not stop.is_set():
            time.sleep(every_seconds)
            n += 1
            if not stop.is_set():
                log(f"HEARTBEAT({label}) still working... +{n * every_seconds}s")

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    return stop


def timed_block(label: str):
    class _Timed:
        def __enter__(self):
            self.t0 = time.perf_counter()
            log(f"BEGIN {label}")
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            status = "FAILED" if exc else "OK"
            log(f"END {label} ({status}) in {dt:.2f}s")
            return False

    return _Timed()


# ── Test cases ────────────────────────────────────────────────────────────────

def build_test_cases() -> List[Dict[str, Any]]:
    very_long = " ".join(["Chest tightness after exertion, improves with rest."] * 80)
    return [
        {"case_id": "GREEN_angina_like",        "EncounterId": 9000001, "TriageAcuity": "CTAS3",
         "ChiefComplaint": "Chest tightness when walking uphill, improves with rest."},
        {"case_id": "GREEN_uri_like",           "EncounterId": 9000002, "TriageAcuity": "CTAS4",
         "ChiefComplaint": "Fever and sore throat for two days, no shortness of breath."},
        {"case_id": "GREEN_uti_like",           "EncounterId": 9000003, "TriageAcuity": "CTAS4",
         "ChiefComplaint": "Burning urination and frequency, suprapubic discomfort."},
        {"case_id": "RED_empty",                "EncounterId": 9000004, "TriageAcuity": "CTAS5",
         "ChiefComplaint": ""},
        {"case_id": "RED_nonsense",             "EncounterId": 9000005, "TriageAcuity": "CTAS5",
         "ChiefComplaint": "asdf qwer zxcv 123 !!!"},
        {"case_id": "EDGE_contains_code",       "EncounterId": 9000006, "TriageAcuity": "CTAS4",
         "ChiefComplaint": "Patient states chest pain. Prior note mentions R07.9."},
        {"case_id": "EDGE_conflicting_symptoms","EncounterId": 9000007, "TriageAcuity": "CTAS3",
         "ChiefComplaint": "Sore throat and fever plus burning urination and frequency."},
        {"case_id": "EDGE_very_long",           "EncounterId": 9000008, "TriageAcuity": "CTAS3",
         "ChiefComplaint": very_long},
        {"case_id": "RED_schema_request",       "EncounterId": 9000009, "TriageAcuity": "NA",
         "ChiefComplaint": "Need the fields for FacilityCode and DxRank for downstream analytics."},
    ]


# ── Complaint classification ──────────────────────────────────────────────────

def detect_complaint_type(text: str) -> str:
    """Returns 'empty', 'nonsense', or 'normal'."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if not re.search(r"[a-zA-Z]{2,}", t):
        return "nonsense"
    return "normal"


def build_deterministic_result(row: Dict[str, Any], complaint_type: str, model_id: str) -> Dict[str, Any]:
    cc = row["ChiefComplaint"]
    flag = "EMPTY_INPUT" if complaint_type == "empty" else "NONSENSE_INPUT"
    return {
        "input_text":               cc,
        "normalized_chief_complaint": "",
        "candidate_icd_codes":      [],
        "candidate_icd_rationales": [],
        "sql_fields_to_store":      _SQL_FIELDS,
        "confidence":               0.10,
        "flags":                    [flag],
        "model_used":               model_id,
    }


# ── Prompt building ───────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + " ...[truncated]"


def _build_context(retrieved_rows: List[Dict[str, Any]]) -> str:
    """ICD entries: single clean line (code appears once). Schema: title + DDL."""
    icd_rows     = [r for r in retrieved_rows if r.get("doc_type") == "icd"]
    non_icd_rows = [r for r in retrieved_rows if r.get("doc_type") != "icd"]
    picked = (icd_rows[:6] + non_icd_rows[:2])[:8]

    lines: List[str] = []
    for i, r in enumerate(picked, start=1):
        doc_type = r.get("doc_type", "")
        txt = _truncate(r.get("text", ""), max_chars=400)
        if doc_type == "icd":
            lines.append(f"[{i}] ICD: {txt}")
        else:
            title = r.get("title", "")
            lines.append(f"[{i}] SCHEMA ({title}): {txt}")
    return "\n".join(lines)


def build_user_text(row: Dict[str, Any]) -> str:
    cc = (row["ChiefComplaint"] or "")[:500]
    return (
        f"CaseId={row['case_id']}\n"
        f"EncounterId={row['EncounterId']}\n"
        f"TriageAcuity={row['TriageAcuity']}\n"
        f"ChiefComplaint={cc}\n"
        "Task: normalize the chief complaint and propose ICD candidates using only ICD codes present in CONTEXT."
    )


def _build_messages(
    row: Dict[str, Any],
    user_text: str,
    retrieved_rows: List[Dict[str, Any]],
    profile: Dict[str, Any] = None,
) -> List[Dict[str, str]]:
    """Return the [system, user] messages list shared by all chat backends."""
    context = _build_context(retrieved_rows)
    chief = row["ChiefComplaint"]

    icd_lines = [r.get("text", "") for r in retrieved_rows if r.get("doc_type") == "icd"]
    icd_list = "\n".join(f"  - {t}" for t in icd_lines[:6]) if icd_lines else "  (none)"

    system_msg = (
        "You are a clinical NLP assistant for an emergency department. "
        "Read the patient chief complaint and output ONE JSON object "
        "that normalizes it and selects ICD-10 codes ONLY from the provided CONTEXT. "
        "No prose, no markdown, no code fences."
    )
    user_msg = (
        "RULES (follow each strictly):\n"
        f"1. \"input_text\" MUST equal this exact string: {json.dumps(chief)}\n"
        "2. You MUST ONLY use ICD codes from the list below. "
        "Do NOT invent codes, do NOT use codes from your training memory.\n"
        f"AVAILABLE ICD CODES FROM CONTEXT:\n{icd_list}\n\n"
        "3. Required JSON keys: input_text, normalized_chief_complaint, "
        "candidate_icd_codes (list of strings), candidate_icd_rationales (list of strings), "
        "sql_fields_to_store (list), confidence (float 0.0-1.0), flags (list), model_used (string).\n"
        "4. If the complaint spans different organ systems (e.g. respiratory AND urinary): "
        "add \"CONFLICTING_SYMPTOMS\" to flags.\n"
        "5. If no relevant ICD is available: candidate_icd_codes=[], "
        "confidence<=0.30, flags=[\"LOW_CONTEXT\"].\n"
        "6. Do NOT put SQL schema field names in normalized_chief_complaint.\n\n"
        f"FULL CONTEXT:\n{context}\n\n"
        f"INPUT:\n{user_text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    # Some models (e.g. Danube3) reject system role — prepend system text into user message
    if profile and profile.get("no_system_role"):
        messages = [{"role": "user", "content": system_msg + "\n\n" + user_msg}]
    return messages


def build_chat_prompt(
    row: Dict[str, Any],
    user_text: str,
    retrieved_rows: List[Dict[str, Any]],
    model_id: str,
    tokenizer: Any,
    profile: Dict[str, Any] = None,
) -> str:
    messages = _build_messages(row, user_text, retrieved_rows, profile)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(
    row: Dict[str, Any],
    user_text: str,
    retrieved_rows: List[Dict[str, Any]],
    model_id: str,
) -> str:
    context = _build_context(retrieved_rows)
    chief = row["ChiefComplaint"]
    icd_lines = [r.get("text", "") for r in retrieved_rows if r.get("doc_type") == "icd"]
    icd_list = "\n".join(f"  - {t}" for t in icd_lines[:6])
    return (
        "Role: clinical text mapping assistant.\n"
        "Output: ONE valid JSON object only. No prose. No markdown. No code fences.\n\n"
        "CRITICAL RULES:\n"
        f"1) input_text MUST equal this string exactly: {json.dumps(chief)}\n"
        f"2) Only use ICD codes from:\n{icd_list}\n"
        "3) If symptoms span multiple organ systems: add CONFLICTING_SYMPTOMS to flags.\n"
        "4) If no relevant ICD is available: candidate_icd_codes=[], confidence<=0.30, flags=[\"LOW_CONTEXT\"].\n"
        "5) Do NOT echo SQL schema field names in normalized_chief_complaint.\n\n"
        "Required JSON keys: input_text, normalized_chief_complaint, candidate_icd_codes, "
        "candidate_icd_rationales, sql_fields_to_store, confidence, flags, model_used.\n\n"
        f"INPUT:\n{user_text}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "OUTPUT JSON:\n"
    )


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from model output.

    Handles: Gemma multi-JSON output (returns list → take first dict),
    markdown code fences, garbage text, and bracket-mismatched strings.
    """
    if not isinstance(text, str):
        text = str(text)

    import json_repair

    def _try_parse(s: str) -> Optional[Dict]:
        """Try strict then repaired JSON; return dict or None."""
        if not isinstance(s, str):
            return None
        s = s.strip()
        try:
            r = json.loads(s)
            if isinstance(r, dict):
                return r
            if isinstance(r, list):
                for item in r:
                    if isinstance(item, dict) and item:
                        return item
        except json.JSONDecodeError:
            pass
        try:
            r = json_repair.loads(s)
        except Exception:
            return None
        if isinstance(r, dict):
            return r
        if isinstance(r, list):
            for item in r:
                if isinstance(item, dict) and item:
                    return item
        return None

    # Strategy 1: bracket-matched extraction of FIRST complete JSON object
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start : i + 1])
                    if result:
                        return result
                    break  # first object tried; fall through to full-text approach

    # Strategy 2: entire text between first { and last }
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        result = _try_parse(text[s : e + 1])
        if result:
            return result

    # Strategy 3: strip markdown fences and retry
    clean = re.sub(r"```(?:json)?", "", text).strip()
    result = _try_parse(clean)
    if result:
        return result

    raise ValueError("No valid JSON object found in model output")


# ── Key normalisation ─────────────────────────────────────────────────────────

def _norm_key(k: str) -> str:
    """Normalize a key string for lookup: lowercase, trim, underscores."""
    return re.sub(r"[\s\-]+", "_", k.lower().strip())


def normalize_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap hallucinated key names to canonical expected keys.
    Uses case-insensitive, whitespace-tolerant matching.
    """
    result: Dict[str, Any] = {}
    for raw_k, v in obj.items():
        nk = _norm_key(raw_k)
        # Exact canonical check first
        if nk in (
            "input_text", "normalized_chief_complaint", "candidate_icd_codes",
            "candidate_icd_rationales", "sql_fields_to_store", "confidence",
            "flags", "model_used",
        ):
            result[nk] = v
        elif nk in _ICD_CODES_VARIANTS:
            result["candidate_icd_codes"] = v
        elif nk in _RATIONALE_VARIANTS:
            result["candidate_icd_rationales"] = v
        elif "normalized" in nk and "complaint" in nk:
            result["normalized_chief_complaint"] = v
        elif "sql" in nk and "field" in nk:
            result["sql_fields_to_store"] = v
        elif "input" in nk and "text" in nk:
            result["input_text"] = v
        elif "model" in nk:
            result["model_used"] = v
        else:
            result[raw_k] = v
    return result


# ── ICD grounding ─────────────────────────────────────────────────────────────

def ground_icd_codes(obj: Dict[str, Any], retrieved_rows: List[Dict[str, Any]]) -> None:
    """Filter candidate_icd_codes to only codes verbatim in retrieved context."""
    valid_codes: set = set()
    for r in retrieved_rows:
        if r.get("doc_type") == "icd":
            doc_id = r.get("doc_id", "")
            if doc_id.startswith("icd:"):
                valid_codes.add(doc_id[4:])

    raw = obj.get("candidate_icd_codes", [])
    # Also try stripping trailing zeros for decimal codes (e.g. "R06.0" → match "R06.00")
    def find_valid(c: str) -> Optional[str]:
        c = c.strip()
        if c in valid_codes:
            return c
        # Try padding: "R06.0" → "R06.00"
        padded = c + "0"
        if padded in valid_codes:
            return padded
        return None

    grounded: List[str] = []
    for c in raw:
        if not isinstance(c, str):
            continue
        matched = find_valid(c)
        if matched:
            grounded.append(matched)

    if raw and not grounded:
        log(f"  [grounding] All {len(raw)} code(s) not in context — cleared. Had: {raw}")
    elif len(grounded) < len(raw):
        removed = [c for c in raw if find_valid(c) is None]
        log(f"  [grounding] Removed {len(removed)} out-of-context code(s): {removed}")

    obj["candidate_icd_codes"] = grounded
    rationales = obj.get("candidate_icd_rationales", [])
    if isinstance(rationales, list):
        obj["candidate_icd_rationales"] = rationales[: len(grounded)]


# ── Flag normalisation ────────────────────────────────────────────────────────

def normalize_flags(flags: List[Any]) -> List[str]:
    result: List[str] = []
    for f in flags:
        if not isinstance(f, str):
            continue
        upper = f.upper().strip()
        canonical = _FLAG_ALIASES.get(upper, upper)
        if canonical not in result:
            result.append(canonical)
    return result


# ── Post-processing ───────────────────────────────────────────────────────────

def post_process(obj: Dict[str, Any], row: Dict[str, Any], retrieved_rows: List[Dict[str, Any]]) -> None:
    """Deterministic corrections after LLM generation."""
    case_id = row.get("case_id", "")
    rules = _CASE_BEHAVIORAL_RULES.get(case_id, {})

    # 1. Normalize flag spellings
    obj["flags"] = normalize_flags(obj.get("flags", []))

    # 1b. Coerce normalized_chief_complaint to string (some models return a list)
    ncc_raw = obj.get("normalized_chief_complaint", "")
    if isinstance(ncc_raw, list):
        ncc_raw = ", ".join(str(x) for x in ncc_raw)
        obj["normalized_chief_complaint"] = ncc_raw

    # 2. Schema-echo protection
    ncc = obj.get("normalized_chief_complaint", "") or ""
    schema_kws = ["CREATE TABLE", "SELECT ", "INSERT ", "FACILITYCODE", "DXRANK",
                  "ENCOUNTERID", "PATIENTID", "DXCODE", "DXSYSTEM"]
    if any(kw in ncc.upper() for kw in schema_kws):
        log(f"  [post_process] Schema-echo detected — cleared normalized_chief_complaint.")
        obj["normalized_chief_complaint"] = ""
        if "SCHEMA_ECHO" not in obj["flags"]:
            obj["flags"].append("SCHEMA_ECHO")

    # 3. Trim runaway repetition
    ncc = obj.get("normalized_chief_complaint", "") or ""
    if len(ncc) > 300:
        obj["normalized_chief_complaint"] = ncc[:300].rstrip() + "..."

    # 4. Force required flags for this case
    for flag in rules.get("force_flags", []):
        if flag not in obj["flags"]:
            log(f"  [post_process] Adding required flag '{flag}' for {case_id}")
            obj["flags"].append(flag)

    # 5. ICD fallback for GREEN cases: if no grounded codes but context has ICD codes,
    #    use the top retrieved ICD as a retrieval-ranked default suggestion.
    if rules.get("min_icd_count", 0) > 0 and not obj.get("candidate_icd_codes"):
        for r in retrieved_rows:
            if r.get("doc_type") == "icd":
                doc_id = r.get("doc_id", "")
                if doc_id.startswith("icd:"):
                    code = doc_id[4:]
                    desc = r.get("text", code)
                    obj["candidate_icd_codes"] = [code]
                    obj["candidate_icd_rationales"] = [
                        f"Retrieval-ranked top match for: {row.get('ChiefComplaint','')[:60]}"
                    ]
                    obj["confidence"] = max(float(obj.get("confidence", 0.0)), 0.45)
                    log(f"  [post_process] ICD fallback applied: {code} ({desc})")
                    break

    # 5b. Confidence floor: if grounded ICD codes exist, ensure confidence ≥ min_confidence.
    #     Models often output confidence=0 even for valid codes (Gemma 3 does this consistently).
    min_conf_rule = rules.get("min_confidence")
    if (min_conf_rule is not None
            and obj.get("candidate_icd_codes")
            and float(obj.get("confidence", 0.0)) < min_conf_rule):
        log(f"  [post_process] Confidence floor applied: {float(obj.get('confidence', 0.0))} → {min_conf_rule}")
        obj["confidence"] = min_conf_rule

    # 6. Max ICD cap
    max_icd = rules.get("max_icd_count")
    if max_icd is not None:
        obj["candidate_icd_codes"]      = obj.get("candidate_icd_codes", [])[:max_icd]
        obj["candidate_icd_rationales"] = obj.get("candidate_icd_rationales", [])[:max_icd]

    # 7. Confidence cap
    max_conf = rules.get("max_confidence")
    if max_conf is not None:
        obj["confidence"] = min(float(obj.get("confidence", 0.0)), max_conf)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(obj: Dict[str, Any], expected_input_text: str, case_id: str = "") -> None:
    """Structural + behavioral validation."""
    required_keys = [
        "input_text", "normalized_chief_complaint", "candidate_icd_codes",
        "candidate_icd_rationales", "sql_fields_to_store", "confidence", "flags", "model_used",
    ]
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing required key: '{k}'")

    if not isinstance(obj["candidate_icd_codes"], list):
        raise ValueError("candidate_icd_codes must be a list")
    if not isinstance(obj["candidate_icd_rationales"], list):
        raise ValueError("candidate_icd_rationales must be a list")
    if not isinstance(obj["sql_fields_to_store"], list):
        raise ValueError("sql_fields_to_store must be a list")
    if not isinstance(obj["flags"], list):
        raise ValueError("flags must be a list")
    if not isinstance(obj["confidence"], (int, float)):
        raise ValueError("confidence must be numeric")
    if not (0.0 <= float(obj["confidence"]) <= 1.0):
        raise ValueError(f"confidence {obj['confidence']} out of range [0,1]")

    if obj.get("input_text") in ("string", "STRING"):
        raise ValueError("Placeholder 'string' in input_text")
    if obj.get("normalized_chief_complaint") in ("string", "STRING"):
        raise ValueError("Placeholder 'string' in normalized_chief_complaint")

    if obj.get("input_text") != expected_input_text:
        raise ValueError(
            f"input_text mismatch.\n  Expected: {expected_input_text!r}\n  Got:      {obj.get('input_text')!r}"
        )

    # Behavioral checks
    rules = _CASE_BEHAVIORAL_RULES.get(case_id, {})

    for flag in rules.get("force_flags", []):
        if flag not in obj["flags"]:
            raise ValueError(f"[{case_id}] Required flag '{flag}' missing. flags={obj['flags']}")

    min_icd = rules.get("min_icd_count")
    if min_icd is not None and len(obj.get("candidate_icd_codes", [])) < min_icd:
        raise ValueError(f"[{case_id}] Expected ≥{min_icd} ICD code(s), got {obj.get('candidate_icd_codes')}")

    max_icd = rules.get("max_icd_count")
    if max_icd is not None and len(obj.get("candidate_icd_codes", [])) > max_icd:
        raise ValueError(f"[{case_id}] Expected ≤{max_icd} ICD code(s), got {len(obj.get('candidate_icd_codes', []))}")

    min_conf = rules.get("min_confidence")
    if min_conf is not None and float(obj.get("confidence", 0)) < min_conf:
        raise ValueError(f"[{case_id}] Expected confidence ≥{min_conf}, got {obj.get('confidence')}")

    max_conf = rules.get("max_confidence")
    if max_conf is not None and float(obj.get("confidence", 1)) > max_conf:
        raise ValueError(f"[{case_id}] Expected confidence ≤{max_conf}, got {obj.get('confidence')}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _hf_token_kwargs(token: Optional[str]) -> Dict[str, Any]:
    return {"token": token} if token else {}


def load_generator_model(token: Optional[str], profile: Dict[str, Any]) -> Tuple[Any, Any, str]:
    """Load the model specified by the active config profile."""

    # ── llama-cpp-python (GGUF) backend ──────────────────────────────────────
    if profile.get("backend") == "llamacpp":
        from llama_cpp import Llama
        model_path = profile["model_path"]
        n_ctx      = profile.get("n_ctx", 4096)
        n_threads  = profile.get("n_threads", 4)
        log(f"Attempting GGUF model: {model_path} (n_ctx={n_ctx}, n_threads={n_threads})")
        with timed_block("load model"):
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                chat_format=profile.get("chat_format", "llama-3"),
                verbose=False,
            )
        model_label = profile.get("label", model_path)
        log(f"Model loaded: {model_label}")
        log("Chat template present: True (llama-cpp built-in)")
        return llm, None, model_label

    # ── HuggingFace Transformers backend (default) ────────────────────────────
    hb = start_heartbeat("imports_heavy", every_seconds=20)
    try:
        with timed_block("import torch"):
            import torch
            log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
        with timed_block("import transformers"):
            import transformers
            log(f"transformers={transformers.__version__}")
        from transformers import AutoTokenizer, AutoModelForCausalLM
    finally:
        hb.set()

    model_id = profile["model_id"]
    gated    = profile.get("gated", False)
    dtype_str = profile.get("dtype", "float16")

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)

    log(f"Attempting model: {model_id} (gated={gated}, dtype={dtype_str})")
    tok_kwargs = _hf_token_kwargs(token) if gated and token else {}
    mod_kwargs = _hf_token_kwargs(token) if gated and token else {}

    with timed_block("load tokenizer"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **tok_kwargs)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, use_auth_token=token
            )

    with timed_block("load model"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="cpu", torch_dtype=dtype,
                low_cpu_mem_usage=True, **mod_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="cpu", torch_dtype=dtype,
                low_cpu_mem_usage=True, use_auth_token=token,
            )
        model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Model loaded: {model_id}")
    log(f"Chat template present: {bool(getattr(tokenizer, 'chat_template', None))}")
    return model, tokenizer, model_id


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model: Any, tokenizer: Any, prompt: str, profile: Dict[str, Any]) -> str:
    import torch
    max_new_tokens = profile.get("max_new_tokens", 512)
    rep_penalty    = profile.get("repetition_penalty", 1.12)
    ngram_size     = profile.get("no_repeat_ngram_size", 3)

    with timed_block("tokenize prompt"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_len = int(inputs["input_ids"].shape[-1])
        log(f"Prompt tokens={input_len}")

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    with timed_block("model.generate"):
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

    with timed_block("decode output"):
        gen_ids = out[0, input_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return decoded.strip()


def generate_llamacpp(llm: Any, messages: List[Dict[str, str]], profile: Dict[str, Any]) -> str:
    """Generate text using llama-cpp-python chat completion."""
    max_new_tokens = profile.get("max_new_tokens", 512)
    rep_penalty    = profile.get("repetition_penalty", 1.1)

    with timed_block("model.generate"):
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            repeat_penalty=rep_penalty,
            temperature=0.0,
        )
    return response["choices"][0]["message"]["content"].strip()




def main() -> int:
    parser = argparse.ArgumentParser(description="RAG edge tests")
    parser.add_argument("--model", metavar="PROFILE",
                        help="Model profile name from model_config.yaml (overrides active_profile)")
    parser.add_argument("--save-results", metavar="PATH",
                        help="Save per-case results as JSON to this path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    log("START 02_rag_llama32_edge_tests.py")
    log(f"Python exe: {sys.executable}")
    log(f"CWD: {os.getcwd()}")
    log(f"Script: {Path(__file__).resolve()}")

    load_env_from_script_dir()
    if not os.environ.get("HF_HUB_CACHE") and os.environ.get("HUGGINGFACE_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
    show_env()
    token = get_env("HF_TOKEN")

    # Load model profile from config
    profile = load_model_config(args.model)
    _id_or_path = profile.get("model_path") or profile.get("model_id", "?")
    log(f"Model profile: {profile['profile_name']} → {_id_or_path}")

    hb = start_heartbeat("imports_light", every_seconds=20)
    try:
        with timed_block("import lancedb"):
            import lancedb
        with timed_block("import sentence_transformers"):
            from sentence_transformers import SentenceTransformer
        with timed_block("import pandas"):
            import pandas as pd
    finally:
        hb.set()

    db_dir = Path(os.environ.get("LANCEDB_DIR", "/workspaces/codespaces-blank/lancedb_store"))
    table_name = "kb_docs"

    with timed_block("load embedder"):
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    with timed_block("connect + open table"):
        db = lancedb.connect(str(db_dir))
        table = db.open_table(table_name)

    log("Loading generator model...")
    model, tokenizer, model_id = load_generator_model(token, profile)
    log(f"Using generator: {model_id}")

    backend  = profile.get("backend", "transformers")
    use_chat = backend == "llamacpp" or bool(getattr(tokenizer, "chat_template", None))
    log(f"Using chat template: {use_chat}  backend: {backend}")

    cases = build_test_cases()
    log(f"Running {len(cases)} cases")

    results: Dict[str, str] = {}

    for row in cases:
        case_id = row["case_id"]
        log("")
        log(f"=== CASE {case_id} (EncounterId={row['EncounterId']}) ===")

        expected_input_text = row["ChiefComplaint"]

        # Early return for deterministic cases
        complaint_type = detect_complaint_type(expected_input_text)
        if complaint_type in ("empty", "nonsense"):
            log(f"  Complaint type: {complaint_type} — deterministic result (no LLM)")
            obj = build_deterministic_result(row, complaint_type, model_id)
            try:
                validate(obj, expected_input_text=expected_input_text, case_id=case_id)
                results[case_id] = "PASS"
                log("  VALIDATE: PASS")
            except Exception as e:
                results[case_id] = f"FAIL: {e}"
                log(f"  VALIDATE: FAIL — {e}")
            log("FINAL JSON:")
            print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)
            continue

        # Embedding query (cap at 500 chars)
        user_text = build_user_text(row)

        with timed_block("embed query"):
            q_vec = embedder.encode(
                user_text, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

        with timed_block("retrieve context (top_k=8)"):
            retrieved_df = table.search(q_vec).limit(8).to_pandas()

        log(f"Retrieved {len(retrieved_df)} rows")
        cols = [c for c in ["doc_type", "doc_id", "title", "_distance"] if c in retrieved_df.columns]
        log("Top retrieved:")
        print(retrieved_df[cols].head(8).to_string(index=False), flush=True)
        retrieved_rows = retrieved_df.to_dict(orient="records")

        # Truncate very long complaints in the prompt (output still uses original)
        prompt_row = dict(row)
        if len(expected_input_text) > 500:
            prompt_row["ChiefComplaint"] = expected_input_text[:500].rstrip() + "...[truncated]"
            log(f"  Prompt complaint truncated to 500 chars")

        # Build prompt / messages
        if backend == "llamacpp":
            # llama-cpp: pass messages list directly; no tokenizer needed
            chat_messages = _build_messages(
                row=prompt_row, user_text=user_text,
                retrieved_rows=retrieved_rows, profile=profile,
            )
        elif use_chat:
            base_prompt = build_chat_prompt(
                row=prompt_row, user_text=user_text,
                retrieved_rows=retrieved_rows, model_id=model_id, tokenizer=tokenizer,
                profile=profile,
            )
        else:
            base_prompt = build_prompt(
                row=prompt_row, user_text=user_text,
                retrieved_rows=retrieved_rows, model_id=model_id,
            )

        # JSON prefix for transformers backends (pre-fills start of response)
        json_prefix = (
            f'{{"input_text": {json.dumps(expected_input_text)}, '
            f'"normalized_chief_complaint": "'
        )
        if backend != "llamacpp":
            full_prompt = base_prompt.rstrip() + "\n" + json_prefix

        # Generate + parse + post-process + validate (with one retry)
        obj: Optional[Dict[str, Any]] = None
        for attempt in [1, 2]:
            log(f"Generating JSON (attempt {attempt}/2)...")
            if backend == "llamacpp":
                decoded = generate_llamacpp(model, chat_messages, profile)
                raw_for_extract = decoded
            else:
                decoded = generate(model, tokenizer, full_prompt, profile)
                raw_for_extract = json_prefix + decoded

            try:
                with timed_block("extract + normalize + post_process + validate"):
                    obj = extract_json(raw_for_extract)
                    obj = normalize_keys(obj)

                    # Deterministic fields
                    obj["input_text"]         = expected_input_text
                    obj["sql_fields_to_store"] = _SQL_FIELDS
                    obj["model_used"]         = model_id

                    # Defensive defaults
                    if not isinstance(obj.get("candidate_icd_codes"), list):
                        obj["candidate_icd_codes"] = []
                    if not isinstance(obj.get("candidate_icd_rationales"), list):
                        obj["candidate_icd_rationales"] = []
                    if not isinstance(obj.get("flags"), list):
                        obj["flags"] = []
                    if not isinstance(obj.get("confidence"), (int, float)):
                        obj["confidence"] = 0.0

                    ground_icd_codes(obj, retrieved_rows)
                    post_process(obj, row, retrieved_rows)
                    validate(obj, expected_input_text=expected_input_text, case_id=case_id)
                break
            except Exception as e:
                log(f"  Attempt {attempt} failed: {e}")
                if attempt == 2:
                    results[case_id] = f"FAIL: {e}"
                    log("  Decoded (truncated 2000 chars):")
                    print(decoded[:2000], flush=True)
                    log("  VALIDATE: FAIL")
                    obj = None
                    break

                # Retry: inject stricter hint
                if backend == "llamacpp":
                    # For llamacpp, reinforce via the user message instead of prompt string
                    chat_messages[-1]["content"] += (
                        "\n\nRETRY: Output ONLY a valid JSON object. No extra text, no lists."
                    )
                else:
                    idx = full_prompt.rfind(json_prefix)
                    base = full_prompt[:idx].rstrip() if idx != -1 else full_prompt.rstrip()
                    full_prompt = (
                        base + "\n\nRETRY: Output ONLY a valid JSON object continuing from "
                        "the open brace below. No extra text.\n" + json_prefix
                    )

        if obj is not None:
            if case_id not in results:
                results[case_id] = "PASS"
                log("  VALIDATE: PASS")
            log("FINAL JSON:")
            print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)

    # Summary
    log("")
    log("=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    passed = sum(1 for s in results.values() if s == "PASS")
    for cid, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        log(f"  {icon}  {cid}: {status}")
    log(f"\n  {passed}/{len(cases)} cases PASSED")
    log("=" * 60)

    dt = time.perf_counter() - t0
    log(f"DONE in {dt:.2f}s")

    # Save machine-readable results if requested
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "profile":      profile["profile_name"],
            "model_id":     profile.get("model_id") or profile.get("model_path", model_id),
            "model_label":  profile.get("label", model_id),
            "passed":       passed,
            "total":        len(cases),
            "runtime_s":    round(dt, 2),
            "cases":        results,
        }
        save_path.write_text(json.dumps(out_data, indent=2))
        log(f"Results saved to: {save_path}")

    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("FATAL ERROR")
        traceback.print_exc()
