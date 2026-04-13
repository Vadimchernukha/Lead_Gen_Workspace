"""
Apollo Contact Cleaner — core logic.

Adapted from clean_apollo.py for use inside a Streamlit app.
Accepts DataFrames / bytes in, returns DataFrame + log out.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable

import anthropic
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDUSTRIES_FILE = DATA_DIR / "industries_mapping.csv"
TITLE_FILE      = DATA_DIR / "title_mapping.csv"
COMPANY_FILE    = DATA_DIR / "company_name_training.csv"
LLM_CACHE_DIR   = Path(__file__).resolve().parent.parent / "llm_cache"

CLAUDE_MODEL     = "claude-sonnet-4-5-20250929"
BATCH_SIZE       = 75
MAX_CONCURRENT   = 5
MAX_RETRIES      = 5
FEW_SHOT_EXAMPLES = 40

APOLLO_COLUMNS = [
    "First Name", "Last Name", "Title", "Company Name", "Email",
    "# Employees", "Industry", "Person Linkedin Url", "Website",
    "Company Linkedin Url", "City", "State", "Country",
    "Company City", "Company State", "Company Country",
    "Apollo Contact Id", "Apollo Account Id",
]

FINAL_COLUMNS = [
    "Company", "Company_Original", "Website", "Industry", "Country", "State", "City",
    "First name", "Last name", "Title", "Title_Original", "Email",
    "Linkedin Person", "Linkedin Company", "Number of employees",
    "Person Country", "Person State", "Person City",
    "Empty_1", "Empty_2", "Empty_3", "Empty_4", "Empty_5",
    "Email_Domain_Match", "Apollo Contact Id", "Apollo Account Id",
]

_UMLAUT_MAP = str.maketrans({
    "ä": "ae", "Ä": "Ae",
    "ö": "oe", "Ö": "Oe",
    "ü": "ue", "Ü": "Ue",
    "ß": "ss",
    "ø": "oe", "Ø": "Oe",
    "å": "a",  "Å": "A",
    "æ": "ae", "Æ": "Ae",
    "é": "e",  "è": "e",  "ê": "e",  "ë": "e",
    "à": "a",  "â": "a",  "ã": "a",
    "î": "i",  "ï": "i",
    "ô": "o",  "õ": "o",
    "ù": "u",  "û": "u",
    "ç": "c",  "Ç": "C",
    "ñ": "n",  "Ñ": "N",
})


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _transliterate(text: str) -> str:
    return text.translate(_UMLAUT_MAP) if text else text


def _fix_caps(name: str) -> str:
    if not name:
        return name
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return name
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    if upper_ratio < 0.7:
        return name
    result = []
    for word in name.split():
        alpha_only = re.sub(r"[^A-Za-zÀ-ÿ]", "", word)
        if word.isupper() and len(alpha_only) <= 4:
            result.append(word)
        else:
            result.append(word.capitalize())
    return " ".join(result)


def clean_company_text(name: str) -> str:
    return _transliterate(_fix_caps(str(name) if name else ""))


def clean_title_text(title: str) -> str:
    return _transliterate(str(title) if title else "")


# ---------------------------------------------------------------------------
# Step A — Name cleaning
# ---------------------------------------------------------------------------

def _clean_name(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    name = str(val).strip()
    if name.isupper():
        name = name.title()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r" {2,}", " ", name)
    return name.strip()


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df["First Name"] = df["First Name"].apply(_clean_name)
    df["Last Name"]  = df["Last Name"].apply(_clean_name)
    return df


# ---------------------------------------------------------------------------
# Step B — Deduplication
# ---------------------------------------------------------------------------

def deduplicate_contacts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    has_email  = df["Email"].notna() & (df["Email"].str.strip() != "")
    email_norm = df["Email"].str.strip().str.lower()
    is_dup     = has_email & email_norm.duplicated(keep="first")
    removed    = int(is_dup.sum())
    df = df[~is_dup].copy().reset_index(drop=True)
    return df, removed


# ---------------------------------------------------------------------------
# Step 1 — Load & filter
# ---------------------------------------------------------------------------

def _normalize_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_and_filter(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8-sig")
    df   = pd.read_csv(io.StringIO(text), dtype=str, low_memory=False)

    target_norm = {_normalize_col(t): t for t in APOLLO_COLUMNS}
    col_rename: dict[str, str] = {}
    for col in df.columns:
        canon = target_norm.get(_normalize_col(col))
        if canon:
            col_rename[col] = canon

    df = df.rename(columns=col_rename)
    for col in APOLLO_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[APOLLO_COLUMNS].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — URL cleaning
# ---------------------------------------------------------------------------

_URL_PREFIX = re.compile(r"^https?://(www\.)?", re.IGNORECASE)


def _clean_url(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    return _URL_PREFIX.sub("", str(val).strip()).rstrip("/")


def clean_urls(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Website", "Person Linkedin Url", "Company Linkedin Url"):
        df[col] = df[col].apply(_clean_url)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Email / domain match
# ---------------------------------------------------------------------------

def _bare_domain(url: str) -> str:
    return url.split("/")[0].lower() if url else ""


def email_domain_match(df: pd.DataFrame) -> pd.DataFrame:
    def _match(row) -> bool:
        email   = str(row["Email"])   if pd.notna(row["Email"])   else ""
        website = str(row["Website"]) if pd.notna(row["Website"]) else ""
        if "@" not in email or not website:
            return False
        email_domain = email.split("@")[-1].lower()
        site_domain  = _bare_domain(website)
        return site_domain in email_domain or email_domain in site_domain

    df["Email_Domain_Match"] = df.apply(_match, axis=1)
    return df


# ---------------------------------------------------------------------------
# Step 4 — Employee bucketing
# ---------------------------------------------------------------------------

_BUCKETS = [
    (1, 10, "1-10"), (11, 50, "11-50"), (51, 100, "51-100"),
    (101, 500, "101-500"), (501, 1000, "501-1000"), (1001, 5000, "1001-5000"),
]


def _bucket(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    nums = re.findall(r"[\d,]+", str(val))
    if not nums:
        return ""
    try:
        n = int(nums[0].replace(",", ""))
    except ValueError:
        return ""
    for lo, hi, label in _BUCKETS:
        if lo <= n <= hi:
            return label
    return "5000+"


def apply_employee_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df["# Employees"] = df["# Employees"].apply(_bucket)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Industry mapping
# ---------------------------------------------------------------------------

def load_industry_mapping() -> dict[str, str]:
    if not INDUSTRIES_FILE.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(INDUSTRIES_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig   = row.get("Original_Industry", "").strip().lower()
            target = row.get("Target_Industry", "").strip()
            if orig:
                mapping[orig] = target
    return mapping


def apply_industry_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    def _map(val):
        if pd.isna(val) or not str(val).strip():
            return val
        return mapping.get(str(val).strip().lower(), val)
    df["Industry"] = df["Industry"].apply(_map)
    return df


# ---------------------------------------------------------------------------
# Step 6 — LLM processing (async, batched, cached)
# ---------------------------------------------------------------------------

def _load_examples(path: Path, orig_col: str, clean_col: str, n: int) -> list[dict]:
    if not path.exists():
        return []
    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            orig  = row.get(orig_col,  "").strip()
            clean = row.get(clean_col, "").strip()
            if orig and clean and orig != clean:
                examples.append({"original": orig, "cleaned": clean})
    return examples


def _build_company_prompt(examples: list[dict]) -> str:
    return f"""You clean company names for B2B cold email personalisation.
Return ONLY the conversational brand name a person would say out loud.

RULES (apply every rule to every input):
1. Strip ALL legal suffixes regardless of language:
   Inc., LLC, Ltd., GmbH, AG, KG, KGaA, ULC, AB, AS, A/S, Oy, OY, B.V., BV,
   N.V., NV, S.A., SA, SAS, SARL, S.r.l., Srl, ApS, Pte., Pvt., Corp., Co.,
   SpA, OÜ, UAB, and any similar suffix. Also "& Co.", "& Co. KG", "& Co. KGaA".
2. Remove geographic qualifiers appended to a brand name (they are NOT part of the brand):
   country names (Germany, France, UK, Ireland, Switzerland, Sweden, Austria,
   Denmark, Finland, Netherlands, Belgium, USA, America…),
   city names (Hamburg, Berlin, Kiel, München, Vienna, London, Chicago…),
   region words (Europe, Northern Europe, EMEA, Nordic, International, Global,
   North America, Western, Eastern, Southern).
   Exception: keep geographic words that ARE the brand (e.g. "American Pan" stays
   "American Pan" because "American" is the brand word, not a qualifier).
3. Remove personal first names or full names that appear before/after the real brand
   (e.g. "Georg Hagelschuer GmbH" → "Hagelschuer",
   "Wilfried Heinzel AG" → "Heinzel").
4. If a word is ALL CAPS and has more than 4 letters, convert to Title Case
   (e.g. WALTERWERK → Walterwerk, PRODITEC → Proditec, VARIOVAC → Variovac).
   Short ALL-CAPS acronyms (≤4 letters) stay uppercase (NTIC, BRP, ZDS, NNZ, PWR).
5. Split fused CamelCase or concatenated words into separate words when the parts
   are common business/technical words
   (e.g. "Schobertechnologies" → "Schober Technologies",
   "ControlTech" → "Control Tech").
   Do NOT split intentional brand stylisations like "AstroNova", "PinMeTo", "knoell".
6. Convert umlauts/diacritics to plain ASCII:
   ö→oe, ü→ue, ä→ae, ß→ss, ø→oe, å→a, æ→ae, é/è/ê→e, ç→c.
7. Preserve stylised lower-case brands (knoell, iPhone).
8. Max 3 words; strip "Group", "Holdings", "Solutions", "Services", "Technologies"
   when they are generic filler and not the brand identity.

Examples (original → cleaned):
{json.dumps(examples, ensure_ascii=False, indent=2)}

INPUT : {{"rows": [{{"key": " ", "company": " "}}, ...]}}
OUTPUT: {{"results": [{{"key": " ", "company": " "}}, ...]}}
Return ONLY the raw JSON. No markdown, no explanation."""


def _build_title_prompt(examples: list[dict]) -> str:
    return f"""You standardise job titles for B2B cold email personalisation.

RULES:
1. Shorten to the most concise, well-known form (target ≤ 4 words).
2. Use standard English abbreviations:
   CEO, CTO, CFO, COO, VP, SVP, EVP, CMO, CRO, CPO, GM,
   Head of, Director of, Manager of.
3. Drop filler qualifiers — "Global", "Regional", "Senior", "Junior",
   "North America", "EMEA", country/city names — UNLESS they change the meaning.
4. Translate non-English titles to English:
   Geschäftsführer → Managing Director, Produktmanager → Product Manager,
   Algemeen directeur → Managing Director, Teknisk Direktör → Technical Director,
   GF → Managing Director, Inhaber → Owner.
5. Resolve slash/combo titles to the most senior role
   (e.g. "VP / CFO / Treasurer" → "CFO").

Examples (original → cleaned):
{json.dumps(examples, ensure_ascii=False, indent=2)}

INPUT : {{"rows": [{{"key": " ", "title": " "}}, ...]}}
OUTPUT: {{"results": [{{"key": " ", "title": " "}}, ...]}}
Return ONLY the raw JSON. No markdown, no explanation."""


def _dedup_companies(df: pd.DataFrame) -> tuple[list[dict], dict[int, str]]:
    row_to_key: dict[int, str] = {}
    key_to_company: dict[str, str] = {}
    for i in df.index:
        domain  = _bare_domain(str(df.at[i, "Website"]) if pd.notna(df.at[i, "Website"]) else "")
        company = str(df.at[i, "Company Name"]) if pd.notna(df.at[i, "Company Name"]) else ""
        key = domain if domain else (f"name:{company.lower().strip()}" if company else f"row:{i}")
        row_to_key[i] = key
        if key not in key_to_company and company:
            key_to_company[key] = company
    return [{"key": k, "company": v} for k, v in key_to_company.items()], row_to_key


def _dedup_titles(df: pd.DataFrame) -> tuple[list[dict], dict[int, str]]:
    row_to_key: dict[int, str] = {}
    key_to_title: dict[str, str] = {}
    for i in df.index:
        title = str(df.at[i, "Title"]) if pd.notna(df.at[i, "Title"]) else ""
        key   = title.lower().strip()
        row_to_key[i] = key
        if key not in key_to_title and title:
            key_to_title[key] = title
    return [{"key": k, "title": v} for k, v in key_to_title.items()], row_to_key


def _cache_path(content_hash: str) -> Path:
    LLM_CACHE_DIR.mkdir(exist_ok=True)
    return LLM_CACHE_DIR / f"{content_hash}.json"


def _load_cache(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"companies": data.get("companies", {}), "titles": data.get("titles", {})}
    return {"companies": {}, "titles": {}}


def _save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _parse_llm_response(raw: str, batch: list[dict]) -> list[dict]:
    try:
        return json.loads(raw).get("results", [])
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group()).get("results", [])
            except Exception:
                pass
    log.error("JSON parse failed; keeping originals for %d items", len(batch))
    return list(batch)


async def _call_with_backoff(client: anthropic.AsyncAnthropic, system: str, user_content: str) -> str:
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            msg = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = delay * (2 ** attempt)
            log.warning("Rate limit – retry %d/%d in %.1fs", attempt + 1, MAX_RETRIES, wait)
            await asyncio.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < MAX_RETRIES - 1:
                wait = delay * (2 ** attempt)
                log.warning("API %d error – retry %d/%d in %.1fs",
                            exc.status_code, attempt + 1, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries")


async def _process_batch(
    client: anthropic.AsyncAnthropic,
    system: str,
    batch: list[dict],
    field: str,
    sem: asyncio.Semaphore,
    subcache: dict[str, str],
    cache: dict,
    cache_path: Path,
    cache_lock: asyncio.Lock,
    progress: dict,
    on_progress: Callable[[float, str], None] | None,
) -> None:
    async with sem:
        raw     = await _call_with_backoff(client, system, json.dumps({"rows": batch}, ensure_ascii=False))
        results = _parse_llm_response(raw, batch)

        async with cache_lock:
            for item in results:
                subcache[item["key"]] = item.get(field, "")
            _save_cache(cache_path, cache)
            progress["done"] += len(results)

        if on_progress:
            pct     = progress["done"] / progress["total"]
            elapsed = time.perf_counter() - progress["t0"]
            eta     = elapsed / progress["done"] * (progress["total"] - progress["done"]) if progress["done"] else 0
            on_progress(pct, f"LLM: {progress['done']}/{progress['total']} — ETA {eta:.0f}s")


async def run_llm_async(
    df: pd.DataFrame,
    api_key: str,
    content_hash: str,
    on_progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    company_ex = _load_examples(COMPANY_FILE, "Company Name for Emails", "Right Company Name", FEW_SHOT_EXAMPLES)
    title_ex   = _load_examples(TITLE_FILE,   "Title",                    "Right Title",         FEW_SHOT_EXAMPLES)

    company_prompt = _build_company_prompt(company_ex)
    title_prompt   = _build_title_prompt(title_ex)

    unique_companies, row_to_company_key = _dedup_companies(df)
    unique_titles,    row_to_title_key   = _dedup_titles(df)

    cp   = _cache_path(content_hash)
    cache = _load_cache(cp)
    company_cache: dict[str, str] = cache["companies"]
    title_cache:   dict[str, str] = cache["titles"]

    pending_companies = [x for x in unique_companies if x["key"] not in company_cache]
    pending_titles    = [x for x in unique_titles    if x["key"] not in title_cache]

    total_pending = len(pending_companies) + len(pending_titles)
    already_done  = (len(unique_companies) - len(pending_companies) +
                     len(unique_titles)    - len(pending_titles))

    if total_pending > 0:
        client     = anthropic.AsyncAnthropic(api_key=api_key)
        sem        = asyncio.Semaphore(MAX_CONCURRENT)
        cache_lock = asyncio.Lock()
        progress   = {
            "done":  already_done,
            "total": len(unique_companies) + len(unique_titles),
            "t0":    time.perf_counter(),
        }

        co_batches = [pending_companies[i: i + BATCH_SIZE] for i in range(0, len(pending_companies), BATCH_SIZE)]
        ti_batches = [pending_titles[i:    i + BATCH_SIZE] for i in range(0, len(pending_titles),    BATCH_SIZE)]

        tasks = [
            _process_batch(client, company_prompt, b, "company", sem,
                           company_cache, cache, cp, cache_lock, progress, on_progress)
            for b in co_batches
        ] + [
            _process_batch(client, title_prompt, b, "title", sem,
                           title_cache, cache, cp, cache_lock, progress, on_progress)
            for b in ti_batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                log.error("Batch %d failed: %s", idx, r)

    df["Company_Clean"] = [
        clean_company_text(company_cache.get(row_to_company_key[i]) or str(df.at[i, "Company Name"] or ""))
        for i in df.index
    ]
    df["Title_Clean"] = [
        clean_title_text(title_cache.get(row_to_title_key[i]) or str(df.at[i, "Title"] or ""))
        for i in df.index
    ]
    return df


# ---------------------------------------------------------------------------
# Step 7 — Rename and reorder
# ---------------------------------------------------------------------------

def rename_and_reorder(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Company"]           = df["Company_Clean"]
    out["Website"]           = df["Website"]
    out["Industry"]          = df["Industry"]
    out["Country"]           = df["Company Country"]
    out["State"]             = df["Company State"]
    out["City"]              = df["Company City"]
    out["First name"]        = df["First Name"]
    out["Last name"]         = df["Last Name"]
    out["Title"]             = df["Title_Clean"]
    out["Email"]             = df["Email"]
    out["Linkedin Person"]   = df["Person Linkedin Url"]
    out["Linkedin Company"]  = df["Company Linkedin Url"]
    out["Number of employees"] = df["# Employees"]
    out["Person Country"]    = df["Country"]
    out["Person State"]      = df["State"]
    out["Person City"]       = df["City"]
    out["Empty_1"]           = ""
    out["Empty_2"]           = ""
    out["Empty_3"]           = ""
    out["Empty_4"]           = ""
    out["Empty_5"]           = ""
    out["Email_Domain_Match"]  = df["Email_Domain_Match"]
    out["Apollo Contact Id"]   = df["Apollo Contact Id"]
    out["Apollo Account Id"]   = df["Apollo Account Id"]
    out["Company_Original"]    = df["Company Name"]
    out["Title_Original"]      = df["Title"]
    return out[FINAL_COLUMNS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    data: bytes,
    api_key: str | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full pipeline.

    Args:
        data:        raw bytes of the uploaded CSV
        api_key:     Anthropic API key (None = skip LLM step)
        on_progress: callback(fraction, message)

    Returns:
        (final_df, log_lines)
    """
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log.info(msg)

    def _prog(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(frac, msg)

    _prog(0.0, "Loading file…")
    df = load_and_filter(data)
    _log(f"Loaded {len(df)} rows × {len(df.columns)} columns.")

    df = clean_names(df)
    _log("Names cleaned.")

    df, removed = deduplicate_contacts(df)
    _log(f"Deduplication: removed {removed} duplicate email(s). {len(df)} rows remaining.")

    _prog(0.1, "Cleaning URLs…")
    df = clean_urls(df)
    _log("URLs cleaned.")

    df = email_domain_match(df)
    _log("Email/domain match column added.")

    df = apply_employee_buckets(df)
    _log("Employee buckets applied.")

    industry_map = load_industry_mapping()
    df = apply_industry_mapping(df, industry_map)
    _log(f"Industry mapping applied ({len(industry_map)} rules).")

    content_hash = hashlib.md5(data).hexdigest()[:12]

    if api_key:
        _prog(0.15, "Starting LLM enrichment…")
        _log("Starting LLM enrichment (async, batched)…")
        try:
            df = asyncio.run(run_llm_async(df, api_key, content_hash, on_progress))
            _log("LLM enrichment complete.")
        except Exception as exc:
            _log(f"LLM step failed: {exc}. Using raw values.")
            df["Title_Clean"]   = df["Title"].fillna("")
            df["Company_Clean"] = df["Company Name"].fillna("")
    else:
        _log("No API key — skipping LLM step, using raw values.")
        df["Title_Clean"]   = df["Title"].fillna("")
        df["Company_Clean"] = df["Company Name"].fillna("")

    _prog(0.95, "Finalising…")
    final = rename_and_reorder(df)
    _log(f"Done. {len(final)} rows → {len(FINAL_COLUMNS)} columns.")
    _prog(1.0, "Complete.")

    return final, log_lines
