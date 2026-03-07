# 04_map_uncoded_problems.py — Map patient problem notes to ICD-9-CM codes via RAG
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.yaml"


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Model config ──────────────────────────────────────────────────────────────

def load_model_config(profile_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    chosen = (
        profile_name
        or os.environ.get("MODEL_PROFILE")
        or cfg.get("active_profile")
    )
    profiles = cfg.get("profiles", {})
    if chosen not in profiles:
        raise ValueError(f"Unknown profile '{chosen}'. Available: {list(profiles.keys())}")
    profile = dict(profiles[chosen])
    profile["profile_name"] = chosen
    return profile


# ── Utility ───────────────────────────────────────────────────────────────────

def get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v else None


def load_env_from_script_dir() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)


def start_heartbeat(label: str, every_seconds: int = 20) -> threading.Event:
    stop = threading.Event()

    def _beat():
        n = 0
        while not stop.is_set():
            time.sleep(every_seconds)
            n += 1
            if not stop.is_set():
                log(f"HEARTBEAT({label}) still working... +{n * every_seconds}s")

    threading.Thread(target=_beat, daemon=True).start()
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


# ── Input data ────────────────────────────────────────────────────────────────

def load_uncoded_problems(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "SyntheticProblemKey":     (row.get("SyntheticProblemKey") or "").strip(),
                "SyntheticPatientKey":     (row.get("SyntheticPatientKey") or "").strip(),
                "PatientProblemDescription": (row.get("PatientProblemDescription") or "").strip(),
            })
    return rows


# ── Complaint classification ──────────────────────────────────────────────────

_VOWELS = frozenset("aeiouAEIOU")

# Pure admin phrases — no clinical content, matched against lowercased stripped text
_ADMIN_EXACT: frozenset = frozenset({
    "care team", "care team:", "unable to contact", "unable to contact pt",
    "unable to contact patient", "patient refused", "pt refused",
    "no show", "chart review", "see chart", "see above", "n/a", "na",
    "follow up", "follow-up", "referral", "referral only",
})
# Admin if the entire text matches this pattern (e.g. "Care Team:" with nothing after)
_ADMIN_TRAILING_COLON_RE = re.compile(r"^[a-z ]+:$", re.IGNORECASE)

# Behavioral/operational complaint language — signals a staff annotation, not a diagnosis
_ADMIN_BEHAVIOR_RE = re.compile(
    r"\b(refused|refusing|ran\s+around|running\s+around|run\s+around|"
    r"run\s+into|running\s+into|exam\s+room|waiting\s+room|reception|"
    r"check.?in|check.?out|no.?show|rescheduled|cancelled|appointment|"
    r"left\s+without\s+(being\s+)?seen|lwbs|eloped|billing|insurance|"
    r"did\s+not\s+(show|arrive|come)|parent\s+refused|guardian\s+refused|"
    r"control\s+child|control\s+children)\b",
    re.IGNORECASE,
)
# Clinical signal words — if ANY of these are present the input is likely clinical
_CLINICAL_SIGNAL_RE = re.compile(
    r"\b(pain|ache|fever|nausea|vomit|cough|bleed|bleed|fracture|"
    r"diabetes|hypertension|htn|cancer|infection|disease|syndrome|disorder|"
    r"injury|wound|burn|rash|swelling|edema|dyspnea|dyspnoea|"
    r"cardiac|renal|hepatic|pulmonary|neuro|ortho|"
    r"diagnosis|diagnos|symptom|chronic|acute|history|hx|"
    r"mg\b|ml\b|mmol|bpm|mmhg|wbc|rbc|hgb|hba1c|bmi|"
    r"nstemi|stemi|oud|nud|copd|afib|chf|ckd|cad|gerd|dvt|pe\b|"
    r"polyp|cyst|mass|nodule|lesion|tumou?r|stenosis|obstruction|"
    r"thalassemi|sickle|obesity|gallbladder|appendix|hernia|"
    r"tonsil|adenoid|nasolacrimal|lacrimal|dermatitis|eczema|"
    r"enuresis|bedwetting|trait|anemia|anaemia)\b",
    re.IGNORECASE,
)

# Retrieval distance above which we consider the context semantically unrelated
# (L2 distance on unit vectors: 0 = identical, √2 ≈ 1.41 = orthogonal)
_LOW_CONTEXT_DISTANCE_THRESHOLD = 1.10


def detect_complaint_type(text: str) -> str:
    """Returns 'empty', 'admin', 'nonsense', or 'normal'."""
    t = (text or "").strip()
    if not t:
        return "empty"

    # Admin: exact match or bare label ending with colon
    tl = t.lower().strip()
    if tl in _ADMIN_EXACT:
        return "admin"
    if _ADMIN_TRAILING_COLON_RE.match(tl) and len(tl.split()) <= 4:
        return "admin"

    # Admin: behavioral/operational complaint with no clinical content
    if _ADMIN_BEHAVIOR_RE.search(t) and not _CLINICAL_SIGNAL_RE.search(t):
        return "admin"

    words = re.findall(r"[a-zA-Z]+", t)
    if not words:
        return "nonsense"

    def _is_plausible(w: str) -> bool:
        wl = w.lower()
        vowel_count = sum(1 for c in wl if c in _VOWELS)
        if vowel_count >= 2:
            return True
        if len(wl) >= 5 and vowel_count >= 1:
            return True
        return False

    if not any(_is_plausible(w) for w in words):
        return "nonsense"
    return "normal"


# ICD-9-CM procedure codes are Volume 3: two leading digits + optional decimal
# e.g. 36.07, 45.42, 51, 01H — recognisable by a 1-or-2-digit numeric prefix.
_PROC_CODE_RE = re.compile(r"^\d{1,2}(\.\d+)?$")
_PROCEDURE_KEYWORDS_RE = re.compile(
    r"\b(surgery|surgical|procedure|operation|repair|resection|excision|"
    r"implant|stent|transplant|bypass|graft|biopsy|catheter|ablation|"
    r"incision|drainage|ligation|amputation|insertion|removal|replacement|"
    r"angioplasty|pci|cabg|pacemaker|dialysis|infusion)\b",
    re.IGNORECASE,
)


def _strip_procedure_codes(
    codes: List[str],
    rationales: List[str],
    problem_desc: str,
) -> tuple:
    """Remove ICD-9-CM procedure codes unless the input explicitly mentions a procedure."""
    if _PROCEDURE_KEYWORDS_RE.search(problem_desc):
        return codes, rationales  # procedures mentioned — keep all
    filtered_codes, filtered_rationales = [], []
    for code, rat in zip(codes, rationales):
        if _PROC_CODE_RE.match(code.strip()):
            pass  # drop procedure code
        else:
            filtered_codes.append(code)
            filtered_rationales.append(rat)
    return filtered_codes, filtered_rationales


# ── Context building ──────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_chars else t[:max_chars].rstrip() + " ...[truncated]"


def _build_context(retrieved_rows: List[Dict[str, Any]]) -> str:
    icd_rows = [r for r in retrieved_rows if r.get("doc_type") == "icd"]
    picked = icd_rows[:8]
    lines: List[str] = []
    for i, r in enumerate(picked, start=1):
        txt = _truncate(r.get("text", ""), max_chars=300)
        lines.append(f"[{i}] ICD: {txt}")
    return "\n".join(lines)


# ── Prompt building ───────────────────────────────────────────────────────────

def _build_messages(
    problem_desc: str,
    retrieved_rows: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    context  = _build_context(retrieved_rows)
    icd_lines = [r.get("text", "") for r in retrieved_rows if r.get("doc_type") == "icd"]
    icd_list  = "\n".join(f"  - {t}" for t in icd_lines[:8]) if icd_lines else "  (none)"

    system_msg = (
        "You are a clinical coding assistant. "
        "Read the patient problem description and output ONE JSON object "
        "selecting ICD-9-CM DIAGNOSIS codes ONLY from the provided CONTEXT. "
        "No prose, no markdown, no code fences."
    )
    user_msg = (
        "RULES (follow each strictly):\n"
        f"1. \"input_text\" MUST equal this exact string: {json.dumps(problem_desc)}\n"
        "2. You MUST ONLY use ICD-9-CM codes from the list below. "
        "Do NOT invent codes, do NOT use codes from your training memory.\n"
        f"AVAILABLE ICD-9-CM CODES FROM CONTEXT:\n{icd_list}\n\n"
        "3. Required JSON keys: input_text, normalized_chief_complaint, "
        "candidate_icd_codes (list of strings), candidate_icd_rationales (list of strings), "
        "candidate_icd_confidences (list of floats 0.0–1.0, one per code), "
        "confidence (float 0.0–1.0, overall), flags (list of strings), model_used (string).\n"
        "4. NORMALIZATION: normalized_chief_complaint MUST be a concise clinical phrase "
        "(2–5 words). Preserve medical terminology exactly as written. Strip only "
        "provider names (e.g. 'Dr. Smith'), clinic names, and years. "
        "Do NOT narrate, do NOT expand, do NOT infer symptoms not stated. "
        "Example: 'Obesity (Dr. Smith) 2022' → 'obesity'. "
        "Example: 'Gallbladder polyp' → 'gallbladder polyp'.\n"
        "5. CODE SELECTION: Select the 1–3 DIAGNOSIS codes that most directly name the "
        "exact condition stated. Do NOT add codes for conditions the patient is merely "
        "'at risk for', potential complications, or comorbidities not mentioned. "
        "Do NOT include procedure codes (e.g. 36.07, 45.42) unless the input explicitly "
        "states that a procedure was performed.\n"
        "6. candidate_icd_rationales MUST have exactly the same length as candidate_icd_codes. "
        "One plain-string explanation per code citing the specific term in the input that "
        "matches. Never nested objects.\n"
        "7. Add 'LOW_CONTEXT' to flags ONLY if candidate_icd_codes is empty.\n\n"
        f"FULL CONTEXT:\n{context}\n\n"
        f"INPUT:\n{problem_desc}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    if profile and profile.get("no_system_role"):
        messages = [{"role": "user", "content": system_msg + "\n\n" + user_msg}]
    return messages


def build_chat_prompt(
    problem_desc: str,
    retrieved_rows: List[Dict[str, Any]],
    tokenizer: Any,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    messages = _build_messages(problem_desc, retrieved_rows, profile)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_plain_prompt(
    problem_desc: str,
    retrieved_rows: List[Dict[str, Any]],
) -> str:
    context   = _build_context(retrieved_rows)
    icd_lines = [r.get("text", "") for r in retrieved_rows if r.get("doc_type") == "icd"]
    icd_list  = "\n".join(f"  - {t}" for t in icd_lines[:8])
    return (
        "Role: clinical coding assistant (ICD-9-CM).\n"
        "Output: ONE valid JSON object only. No prose. No markdown. No code fences.\n\n"
        "CRITICAL RULES:\n"
        f"1) input_text MUST equal this string exactly: {json.dumps(problem_desc)}\n"
        f"2) Only use ICD-9-CM codes from:\n{icd_list}\n"
        "3) normalized_chief_complaint: concise clinical phrase (2–5 words). "
        "Preserve medical terms verbatim. Strip only provider names and dates. "
        "Do NOT narrate or add information not in the input.\n"
        "4) Select 1–3 DIAGNOSIS codes that most directly name the stated condition. "
        "Do NOT add risk-factor or comorbidity codes not stated. "
        "Do NOT include procedure codes unless the input explicitly describes a procedure.\n"
        "5) candidate_icd_rationales MUST have the same length as candidate_icd_codes — "
        "one plain-string rationale per code citing the matching input term.\n"
        "6) Add LOW_CONTEXT to flags ONLY when candidate_icd_codes is empty.\n\n"
        "Required JSON keys: input_text, normalized_chief_complaint, candidate_icd_codes, "
        "candidate_icd_rationales, candidate_icd_confidences (list of floats, one per code), "
        "confidence, flags, model_used.\n\n"
        f"PROBLEM: {problem_desc}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "OUTPUT JSON:\n"
    )


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = str(text)

    import json_repair

    def _try_parse(s: str) -> Optional[Dict]:
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

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start: i + 1])
                    if result:
                        return result
                    break

    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        result = _try_parse(text[s: e + 1])
        if result:
            return result

    clean = re.sub(r"```(?:json)?", "", text).strip()
    result = _try_parse(clean)
    if result:
        return result

    raise ValueError("No valid JSON object found in model output")


# ── Key normalisation ─────────────────────────────────────────────────────────

_ICD_CODES_VARIANTS = {
    "candidate_icd_codes", "candidate_icds", "candidate_icd", "icds", "icd_codes",
    "candidate_icald_codes", "icd", "codes",
}
_RATIONALE_VARIANTS = {
    "candidate_icd_rationales", "candidate_icd_rationale", "icd_rationales",
    "rational_candidate_icd_code", "rationales", "icd_rationale",
}
_CONFIDENCE_LIST_VARIANTS = {
    "candidate_icd_confidences", "candidate_icd_confidence", "icd_confidences",
    "code_confidences", "per_code_confidence", "per_code_confidences",
    "confidences",
}


def _norm_key(k: str) -> str:
    return re.sub(r"[\s\-]+", "_", k.lower().strip())


def normalize_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_k, v in obj.items():
        nk = _norm_key(raw_k)
        if nk in (
            "input_text", "normalized_chief_complaint", "candidate_icd_codes",
            "candidate_icd_rationales", "candidate_icd_confidences",
            "confidence", "flags", "model_used",
        ):
            result[nk] = v
        elif nk in _ICD_CODES_VARIANTS:
            result["candidate_icd_codes"] = v
        elif nk in _RATIONALE_VARIANTS:
            result["candidate_icd_rationales"] = v
        elif nk in _CONFIDENCE_LIST_VARIANTS:
            result["candidate_icd_confidences"] = v
        elif "normalized" in nk and "complaint" in nk:
            result["normalized_chief_complaint"] = v
        elif "input" in nk and "text" in nk:
            result["input_text"] = v
        elif "model" in nk:
            result["model_used"] = v
        else:
            result[raw_k] = v
    return result


# ── ICD grounding ─────────────────────────────────────────────────────────────

def ground_icd_codes(obj: Dict[str, Any], retrieved_rows: List[Dict[str, Any]]) -> None:
    valid_codes: set = set()
    for r in retrieved_rows:
        if r.get("doc_type") == "icd":
            doc_id = r.get("doc_id", "")
            if doc_id.startswith("icd:"):
                valid_codes.add(doc_id[4:])

    raw = obj.get("candidate_icd_codes", [])

    def find_valid(c: str) -> Optional[str]:
        c = c.strip()
        if c in valid_codes:
            return c
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

_FLAG_ALIASES: Dict[str, str] = {
    "CONFLICTINGSYMPTOMS":    "CONFLICTING_SYMPTOMS",
    "CONFLICTING_SYMPTOM":    "CONFLICTING_SYMPTOMS",
    "CONFLICTING_SYMPTOMES":  "CONFLICTING_SYMPTOMS",
    "EMPTY":                  "EMPTY_INPUT",
    "EMPTY_COMPLAINT":        "EMPTY_INPUT",
    "NONSENSE":               "NONSENSE_INPUT",
    "NONSENSE_COMPLAINT":     "NONSENSE_INPUT",
    "LOW_CONFIDENCE":         "LOW_CONTEXT",
    "INSUFFICIENT_CONTEXT":   "LOW_CONTEXT",
    "INSUFFICIENT":           "LOW_CONTEXT",
    "NO_CONTEXT":             "LOW_CONTEXT",
}


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

def post_process(
    obj: Dict[str, Any],
    problem_desc: str,
    retrieved_rows: List[Dict[str, Any]],
) -> None:
    # 0. Nested-JSON recovery: some models (Danube3, Gemma) embed the entire response
    #    dict as a string inside normalized_chief_complaint, leaving candidate_icd_codes=[].
    #    Detect and recover the real data before any other step.
    ncc_str = obj.get("normalized_chief_complaint", "")
    if (not obj.get("candidate_icd_codes")
            and isinstance(ncc_str, str)
            and ("candidate_icd_codes" in ncc_str or "icd_codes" in ncc_str)):
        import ast as _ast
        nested: Optional[Dict] = None
        try:
            nested = _ast.literal_eval(ncc_str.strip())
        except Exception:
            pass
        if not isinstance(nested, dict):
            try:
                nested = json.loads(ncc_str.strip())
            except Exception:
                pass
        if isinstance(nested, dict) and nested.get("candidate_icd_codes"):
            log(f"  [post_process] Recovering nested JSON from normalized_chief_complaint")
            for k, v in nested.items():
                if k == "normalized_chief_complaint":
                    continue
                if k not in obj or (k == "candidate_icd_codes" and not obj[k]):
                    obj[k] = v
            obj["normalized_chief_complaint"] = ""

    # 1. Normalize flag spellings (alias mapping + dedup)
    obj["flags"] = normalize_flags(obj.get("flags", []))

    # Strip CONFLICTING_SYMPTOMS — model training residue; not a valid output flag here
    if "CONFLICTING_SYMPTOMS" in obj["flags"]:
        obj["flags"].remove("CONFLICTING_SYMPTOMS")

    # Strip model-generated NONSENSE_INPUT — admin detection is handled deterministically
    # before the LLM is called; if the model fires this on a clinical input it is wrong.
    if "NONSENSE_INPUT" in obj["flags"] and obj.get("candidate_icd_codes"):
        obj["flags"].remove("NONSENSE_INPUT")
        log(f"  [post_process] Stripped spurious NONSENSE_INPUT (codes present — not admin)")

    # Strip LOW_CONTEXT if codes are present (prompt rule enforcement)
    if "LOW_CONTEXT" in obj["flags"] and obj.get("candidate_icd_codes"):
        obj["flags"].remove("LOW_CONTEXT")
        log(f"  [post_process] Stripped spurious LOW_CONTEXT (codes present)")

    # 1b. Coerce normalized_chief_complaint to string (some models return a list)
    ncc = obj.get("normalized_chief_complaint", "")
    if isinstance(ncc, list):
        ncc = ", ".join(str(x) for x in ncc)
        obj["normalized_chief_complaint"] = ncc

    # 2. JSON-bleed / runaway repetition protection
    ncc = obj.get("normalized_chief_complaint", "") or ""
    if ncc.lstrip().startswith("{") or "candidate_icd_codes" in ncc:
        log(f"  [post_process] JSON-bleed detected in normalized_chief_complaint — cleared")
        ncc = ""
    if len(ncc) > 300:
        ncc = ncc[:300].rstrip() + "..."
    obj["normalized_chief_complaint"] = ncc

    # 3. ICD fallback: if model returned no codes, pick closest retrieval hit
    if not obj.get("candidate_icd_codes"):
        icd_rows = sorted(
            [r for r in retrieved_rows if r.get("doc_type") == "icd"
             and r.get("doc_id", "").startswith("icd:")],
            key=lambda r: r.get("_distance", 9.0),
        )
        if icd_rows:
            best = icd_rows[0]
            code = best["doc_id"][4:]
            desc = best.get("text", code)
            obj["candidate_icd_codes"] = [code]
            obj["candidate_icd_rationales"] = [
                f"Retrieval-ranked top match for: {problem_desc[:60]}"
            ]
            obj["confidence"] = max(float(obj.get("confidence", 0.0)), 0.40)
            log(f"  [post_process] ICD fallback: {code} ({desc})")

    # Universal confidence floor
    if obj.get("candidate_icd_codes") and float(obj.get("confidence", 0.0)) < 0.10:
        obj["confidence"] = 0.10

    # Coerce rationale items
    rationales_raw = obj.get("candidate_icd_rationales", [])
    coerced = []
    for item in rationales_raw:
        if isinstance(item, dict):
            extracted = (
                item.get("rationale") or item.get("explanation")
                or next(iter(item.values()), None)
            )
            coerced.append(str(extracted) if extracted is not None else "No rationale.")
        else:
            coerced.append(item)
    obj["candidate_icd_rationales"] = coerced

    # Align rationale list length to codes list
    codes      = list(obj.get("candidate_icd_codes", []) or [])
    rationales = list(obj.get("candidate_icd_rationales", []) or [])
    if len(rationales) < len(codes):
        rationales = rationales + ["No rationale provided."] * (len(codes) - len(rationales))
    elif len(rationales) > len(codes):
        rationales = rationales[: len(codes)]
    obj["candidate_icd_rationales"] = rationales

    # Strip ICD-9 procedure codes when the input doesn't describe a procedure
    codes, rationales = _strip_procedure_codes(codes, rationales, problem_desc)
    if len(codes) < len(obj.get("candidate_icd_codes", [])):
        removed = set(obj["candidate_icd_codes"]) - set(codes)
        log(f"  [post_process] Stripped procedure codes (no procedure keyword in input): {removed}")
    obj["candidate_icd_codes"]      = codes
    obj["candidate_icd_rationales"] = rationales

    # Per-code retrieval similarity: derived purely from LanceDB L2 distance
    # L2 on unit vectors: 0 = identical, √2 ≈ 1.41 = orthogonal → maps cleanly to [0, 1]
    dist_map: Dict[str, float] = {}
    for r in retrieved_rows:
        c = r.get("icd_code", "")
        d = r.get("_distance", None)
        if c and d is not None:
            dist_map[c] = round(max(0.0, 1.0 - float(d) / 1.41), 3)

    per_code_similarities: List[str] = []
    for code in codes:
        sim = dist_map.get(code, None)
        per_code_similarities.append(str(sim) if sim is not None else "")

    obj["candidate_icd_similarities"] = per_code_similarities

    # Per-code model confidence: what the model itself reported, validated only — no fallback
    raw_conf_list = obj.get("candidate_icd_confidences", [])
    if not isinstance(raw_conf_list, list):
        raw_conf_list = []

    per_code_model_confs: List[str] = []
    for i, code in enumerate(codes):
        if i < len(raw_conf_list):
            try:
                val = float(raw_conf_list[i])
                per_code_model_confs.append(str(round(max(0.0, min(1.0, val)), 3)))
                continue
            except (TypeError, ValueError):
                pass
        per_code_model_confs.append("")   # model didn't report — leave blank, don't fabricate

    obj["candidate_icd_confidences"] = per_code_model_confs


# ── Model loading ─────────────────────────────────────────────────────────────

def load_generator_model(token: Optional[str], profile: Dict[str, Any]) -> Tuple[Any, Any, str]:
    if profile.get("backend") == "llamacpp":
        from llama_cpp import Llama
        model_path = profile["model_path"]
        n_ctx      = profile.get("n_ctx", 4096)
        n_threads  = profile.get("n_threads", 4)
        log(f"Loading GGUF model: {model_path}")
        with timed_block("load model"):
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                chat_format=profile.get("chat_format", "llama-3"),
                verbose=False,
            )
        return llm, None, profile.get("label", model_path)

    hb = start_heartbeat("import_heavy", every_seconds=20)
    try:
        with timed_block("import torch"):
            import torch
        with timed_block("import transformers"):
            import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
    finally:
        hb.set()

    model_id  = profile["model_id"]
    gated     = profile.get("gated", False)
    dtype_str = profile.get("dtype", "float16")
    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)

    tok_kw = {"token": token} if gated and token else {}
    mod_kw = {"token": token} if gated and token else {}

    with timed_block("load tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **tok_kw)
    with timed_block("load model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype=dtype,
            low_cpu_mem_usage=True, **mod_kw,
        )
        model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_id


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model: Any, tokenizer: Any, prompt: str, profile: Dict[str, Any]) -> str:
    import torch
    max_new_tokens = profile.get("max_new_tokens", 512)
    rep_penalty    = profile.get("repetition_penalty", 1.12)
    ngram_size     = profile.get("no_repeat_ngram_size", 3)
    with timed_block("tokenize prompt"):
        inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
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


# ── CSV output ────────────────────────────────────────────────────────────────

_OUTPUT_COLS = [
    "SyntheticProblemKey",
    "SyntheticPatientKey",
    "PatientProblemDescription",
    "NormalizedDescription",
    "CandidateICD9Codes",
    "CandidateICD9Descs",
    "CandidateICD9ModelConf",
    "CandidateICD9Similarity",
    "Rationale",
    "Confidence",
    "Flags",
    "ModelUsed",
]


def _lookup_descs(codes: List[str], retrieved_rows: List[Dict[str, Any]]) -> List[str]:
    """Return descriptions for each code from retrieved context (best-effort)."""
    code_to_desc: Dict[str, str] = {}
    for r in retrieved_rows:
        if r.get("doc_type") == "icd":
            c = r.get("icd_code", "")
            d = r.get("icd_desc", "")
            if c and d:
                code_to_desc[c] = d
    return [code_to_desc.get(c, "") for c in codes]


def write_results_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log(f"CSV written: {out_path}  ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    _default_out = str(script_dir / "icd9_mapping_results.csv")

    parser = argparse.ArgumentParser(description="Map patient problems to ICD-9-CM via RAG")
    parser.add_argument("--model", metavar="PROFILE",
                        help="Model profile from model_config.yaml")
    parser.add_argument("--input", metavar="PATH",
                        default=str(script_dir / "mock_uncoded.csv"),
                        help="Input CSV (default: mock_uncoded.csv)")
    parser.add_argument("--output", metavar="PATH",
                        default=_default_out,
                        help=f"Output CSV (default: {_default_out})")
    args = parser.parse_args()

    t0 = time.perf_counter()
    log("START 04_map_uncoded_problems.py")
    log(f"Python exe: {sys.executable}")
    log(f"CWD: {os.getcwd()}")

    load_env_from_script_dir()
    token = get_env("HF_TOKEN")

    profile = load_model_config(args.model)
    _id_or_path = profile.get("model_path") or profile.get("model_id", "?")
    log(f"Model profile: {profile['profile_name']} → {_id_or_path}")

    hb = start_heartbeat("imports_light", every_seconds=20)
    try:
        with timed_block("import lancedb"):
            import lancedb
        with timed_block("import sentence_transformers"):
            from sentence_transformers import SentenceTransformer
    finally:
        hb.set()

    db_dir     = Path(os.environ.get("LANCEDB_DIR", str(script_dir / "lancedb_store")))
    table_name = "kb_docs_icd9"

    with timed_block("load embedder"):
        embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device="cpu")

    with timed_block("connect + open table"):
        db    = lancedb.connect(str(db_dir))
        table = db.open_table(table_name)

    log("Loading generator model...")
    model, tokenizer, model_id = load_generator_model(token, profile)
    log(f"Using generator: {model_id}")

    backend  = profile.get("backend", "transformers")
    use_chat = backend == "llamacpp" or bool(getattr(tokenizer, "chat_template", None))
    log(f"Backend: {backend}  use_chat_template: {use_chat}")

    # Load input cases
    input_path = Path(args.input)
    cases = load_uncoded_problems(input_path)
    log(f"Loaded {len(cases)} problem rows from {input_path}")

    output_rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        prob_key    = case["SyntheticProblemKey"]
        pat_key     = case["SyntheticPatientKey"]
        problem_desc = case["PatientProblemDescription"]

        log("")
        log(f"=== CASE {idx}/{len(cases)}  ProblemKey={prob_key} ===")
        log(f"  Description: {problem_desc[:120]}")

        complaint_type = detect_complaint_type(problem_desc)

        if complaint_type in ("empty", "nonsense", "admin"):
            log(f"  Complaint type: {complaint_type} — deterministic result (no LLM)")
            flag = "EMPTY_INPUT" if complaint_type == "empty" else "NONSENSE_INPUT"
            output_rows.append({
                "SyntheticProblemKey":     prob_key,
                "SyntheticPatientKey":     pat_key,
                "PatientProblemDescription": problem_desc,
                "NormalizedDescription":   "",
                "CandidateICD9Codes":      "",
                "CandidateICD9Descs":      "",
                "CandidateICD9ModelConf":  "",
                "CandidateICD9Similarity": "",
                "Rationale":               "",
                "Confidence":              "0.10",
                "Flags":                   flag,
                "ModelUsed":               model_id,
            })
            continue

        # Embed problem description for retrieval
        embed_text = problem_desc[:500]
        with timed_block("embed query"):
            q_vec = embedder.encode(
                embed_text, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

        with timed_block("retrieve context (top_k=8)"):
            retrieved_df  = table.search(q_vec).limit(8).to_pandas()

        log(f"  Retrieved {len(retrieved_df)} rows")
        retrieved_rows = retrieved_df.to_dict(orient="records")

        # Distance gate: if ALL retrieved neighbors are semantically distant, skip LLM
        if "_distance" in retrieved_df.columns:
            min_dist = float(retrieved_df["_distance"].min())
            log(f"  Min retrieval distance: {min_dist:.4f}")
            if min_dist > _LOW_CONTEXT_DISTANCE_THRESHOLD:
                log(f"  Distance {min_dist:.4f} > threshold {_LOW_CONTEXT_DISTANCE_THRESHOLD} "
                    f"— LOW_CONTEXT bypass (no LLM)")
                output_rows.append({
                    "SyntheticProblemKey":     prob_key,
                    "SyntheticPatientKey":     pat_key,
                    "PatientProblemDescription": problem_desc,
                    "NormalizedDescription":   problem_desc.strip(),
                    "CandidateICD9Codes":      "",
                    "CandidateICD9Descs":      "",
                    "CandidateICD9ModelConf":  "",
                "CandidateICD9Similarity": "",
                    "Rationale":               "",
                    "Confidence":              "0.10",
                    "Flags":                   "LOW_CONTEXT",
                    "ModelUsed":               model_id,
                })
                continue

        # Build prompt
        prompt_desc = problem_desc[:500].rstrip() + ("...[truncated]" if len(problem_desc) > 500 else "")

        if backend == "llamacpp":
            chat_messages = _build_messages(prompt_desc, retrieved_rows, profile)
        elif use_chat:
            base_prompt = build_chat_prompt(prompt_desc, retrieved_rows, tokenizer, profile)
        else:
            base_prompt = build_plain_prompt(prompt_desc, retrieved_rows)

        json_prefix = (
            f'{{"input_text": {json.dumps(problem_desc)}, '
            f'"normalized_chief_complaint": "'
        )
        if backend != "llamacpp":
            full_prompt = base_prompt.rstrip() + "\n" + json_prefix

        obj: Optional[Dict[str, Any]] = None
        for attempt in [1, 2]:
            log(f"  Generating JSON (attempt {attempt}/2)...")
            try:
                if backend == "llamacpp":
                    decoded = generate_llamacpp(model, chat_messages, profile)
                    raw_for_extract = decoded
                else:
                    decoded = generate(model, tokenizer, full_prompt, profile)
                    raw_for_extract = json_prefix + decoded

                obj = extract_json(raw_for_extract)
                obj = normalize_keys(obj)

                # Deterministic overrides
                obj["input_text"]  = problem_desc
                obj["model_used"]  = model_id

                if not isinstance(obj.get("candidate_icd_codes"), list):
                    obj["candidate_icd_codes"] = []
                if not isinstance(obj.get("candidate_icd_rationales"), list):
                    obj["candidate_icd_rationales"] = []
                if not isinstance(obj.get("flags"), list):
                    obj["flags"] = []
                if not isinstance(obj.get("confidence"), (int, float)):
                    obj["confidence"] = 0.0

                ground_icd_codes(obj, retrieved_rows)
                post_process(obj, problem_desc, retrieved_rows)
                break

            except Exception as e:
                log(f"  Attempt {attempt} failed: {e}")
                if attempt == 2:
                    log(f"  Decoded (first 1000 chars): {decoded[:1000]}")
                    obj = None
                    break
                # Retry with stricter hint
                if backend == "llamacpp":
                    chat_messages[-1]["content"] += (
                        "\n\nRETRY: Output ONLY a valid JSON object. No extra text."
                    )
                else:
                    idx_hint = full_prompt.rfind(json_prefix)
                    base = full_prompt[:idx_hint].rstrip() if idx_hint != -1 else full_prompt.rstrip()
                    full_prompt = (
                        base + "\n\nRETRY: Output ONLY a valid JSON object continuing "
                        "from the open brace below.\n" + json_prefix
                    )

        if obj is None:
            log(f"  FAILED — writing empty result for {prob_key}")
            output_rows.append({
                "SyntheticProblemKey":     prob_key,
                "SyntheticPatientKey":     pat_key,
                "PatientProblemDescription": problem_desc,
                "NormalizedDescription":   "",
                "CandidateICD9Codes":      "",
                "CandidateICD9Descs":      "",
                "CandidateICD9ModelConf":  "",
                "CandidateICD9Similarity": "",
                "Rationale":               "",
                "Confidence":              "0.0",
                "Flags":                   "GENERATION_ERROR",
                "ModelUsed":               model_id,
            })
        else:
            codes       = obj.get("candidate_icd_codes", [])
            rationales  = obj.get("candidate_icd_rationales", [])
            model_confs = obj.get("candidate_icd_confidences", [])
            similarities = obj.get("candidate_icd_similarities", [])
            descs       = _lookup_descs(codes, retrieved_rows)
            log(f"  Codes: {codes}  Confidence: {obj.get('confidence')}")
            log(f"  Model per-code conf: {model_confs}  Similarity: {similarities}")
            output_rows.append({
                "SyntheticProblemKey":      prob_key,
                "SyntheticPatientKey":      pat_key,
                "PatientProblemDescription": problem_desc,
                "NormalizedDescription":    obj.get("normalized_chief_complaint", ""),
                "CandidateICD9Codes":       "|".join(codes),
                "CandidateICD9Descs":       "|".join(descs),
                "CandidateICD9ModelConf":   "|".join(str(c) for c in model_confs),
                "CandidateICD9Similarity":  "|".join(str(s) for s in similarities),
                "Rationale":                " | ".join(rationales),
                "Confidence":               str(obj.get("confidence", "")),
                "Flags":                    "|".join(obj.get("flags", [])),
                "ModelUsed":                model_id,
            })

    # Write CSV
    out_path = Path(args.output)
    write_results_csv(output_rows, out_path)

    dt = time.perf_counter() - t0
    mapped = sum(1 for r in output_rows if r["CandidateICD9Codes"])
    log("")
    log("=" * 60)
    log(f"DONE  {len(output_rows)} cases processed  {mapped} with codes  in {dt:.1f}s")
    log(f"Output: {out_path}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("FATAL ERROR")
        traceback.print_exc()
        raise SystemExit(1)
