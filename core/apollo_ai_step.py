"""Step 2 — Claude API enrichment (company name + title + location)."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Callable

import anthropic

# ---------------------------------------------------------------------------
# Column indices in the Step 1 output schema
# ---------------------------------------------------------------------------
# OUTPUT_COLUMNS from pipeline.py:
# 0  Company Name for Emails
# 1  Website
# 2  Industry
# 3  Employee Range
# 4  Company Country
# 5  Company State
# 6  Company City
# 7  First Name
# 8  Last Name
# 9  Title
# 10 Email
# 11 Person Linkedin Url
# 12 Company Linkedin Url
# 13 Country
# 14 State
# 15 City
# 16 Apollo Contact Id
# 17 Apollo Account Id

COL_COMPANY = 0
COL_WEBSITE = 1
COL_INDUSTRY = 2
COL_EMP_RANGE = 3
COL_COMP_COUNTRY = 4
COL_COMP_STATE = 5
COL_COMP_CITY = 6
COL_FIRST = 7
COL_LAST = 8
COL_TITLE = 9
COL_EMAIL = 10
COL_PERSON_LI = 11
COL_COMPANY_LI = 12
COL_COUNTRY = 13
COL_STATE = 14
COL_CITY = 15

BATCH_SIZE = 20
MODEL = "claude-opus-4-5"
MAX_TRAINING_CHARS = 40_000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TITLE_TRAINING = DATA_DIR / "title_training.csv"
COMPANY_TRAINING = DATA_DIR / "company_name_training.csv"


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------

def _load_training_csv(path: Path, col_in: str, col_out: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        examples = []
        for row in reader:
            inp = row.get(col_in, "").strip()
            out = row.get(col_out, "").strip()
            if inp:
                examples.append({"input": inp, "output": out})
        return examples
    except Exception:
        return []


def _examples_to_text(examples: list[dict[str, str]], label_in: str, label_out: str, max_chars: int) -> str:
    lines: list[str] = []
    total = 0
    for ex in examples:
        line = f'{label_in}: "{ex["input"]}" → {label_out}: "{ex["output"]}"'
        total += len(line) + 1
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)


def _load_examples() -> tuple[str, str]:
    title_ex = _load_training_csv(TITLE_TRAINING, "Title", "Right Title")
    company_ex = _load_training_csv(COMPANY_TRAINING, "Company Name for Emails", "Right Company Name")

    title_text = _examples_to_text(title_ex, "Title", "Right Title", MAX_TRAINING_CHARS)
    company_text = _examples_to_text(company_ex, "Company Name for Emails", "Right Company Name", MAX_TRAINING_CHARS)
    return title_text, company_text


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a B2B data cleaning assistant. You will receive a batch of contact records and must return cleaned values as JSON.

## Company Name Rules (right_company)
- Remove legal suffixes: Inc., LLC, Ltd., GmbH, B.V., AG, S.A., Corp., Co., Pty., PLC — UNLESS they are part of the brand identity
- Fix encoding issues and garbled text
- Apply Title Case
- If input is empty → output empty string

## Title Rules (right_title)
- Canonicalise to standard titles: CEO, CTO, CFO, COO, CMO, CRO, CPO, CISO, VP of Sales, VP of Marketing, VP of Engineering, VP of Product, SVP, EVP, Managing Director, Director, Head of Sales, Head of Marketing, Head of Engineering, Head of Product, Partner, Founder, Co-Founder, President, General Manager, etc.
- NO prefixes like "CXO -", "DIR -", "VP-", "EX-", or parenthetical additions
- Do NOT downgrade seniority: Director ≠ Head of, VP ≠ Director
- If two roles exist, keep the higher-ranking one
- If you cannot determine a canonical title → return empty string

## Location Rules (country / state / city)
- Only fill location fields that are currently EMPTY in the input row
- Use your knowledge of the company/person context to infer location when possible
- Do NOT overwrite existing non-empty values (return them as-is)
- If unknown, return empty string

{company_examples_section}{title_examples_section}
## Output Format
Return ONLY valid JSON, no markdown, no explanation:
{{"results": [{{"i": 0, "right_company": "...", "right_title": "...", "country": "...", "state": "...", "city": "..."}}]}}"""


def _build_system_prompt(title_text: str, company_text: str) -> str:
    company_section = ""
    if company_text:
        company_section = f"## Company Name Examples\n{company_text}\n\n"
    title_section = ""
    if title_text:
        title_section = f"## Title Examples\n{title_text}\n\n"
    return SYSTEM_PROMPT_TEMPLATE.format(
        company_examples_section=company_section,
        title_examples_section=title_section,
    )


def _build_user_message(batch: list[tuple[int, list[str]]]) -> str:
    records = []
    for local_i, (global_i, row) in enumerate(batch):
        records.append({
            "i": local_i,
            "company_name": row[COL_COMPANY],
            "title": row[COL_TITLE],
            "company_country": row[COL_COMP_COUNTRY],
            "company_state": row[COL_COMP_STATE],
            "company_city": row[COL_COMP_CITY],
            "country": row[COL_COUNTRY],
            "state": row[COL_STATE],
            "city": row[COL_CITY],
            "website": row[COL_WEBSITE],
        })
    return json.dumps({"records": records}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------

def run(
    rows: list[list[str]],
    api_key: str,
    on_progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[list[str]], list[str]]:
    """
    Run Step 2 AI enrichment.

    `rows[0]` must be the header row from Step 1 (or equivalent).
    Returns (rows, log) where rows[0] is an updated header.
    """
    log: list[str] = []

    if not rows:
        return rows, ["ERROR: no rows passed to ai_step"]

    header = list(rows[0])
    data_rows = [list(r) for r in rows[1:]]

    if not data_rows:
        return rows, ["No data rows to process."]

    # Insert new columns into header
    title_idx = header.index("Title") if "Title" in header else len(header) - 1
    company_idx = header.index("Company Name for Emails") if "Company Name for Emails" in header else 0

    new_header = list(header)
    # Insert Right Title after Title (offset +1 for company column added before if needed)
    new_header.insert(title_idx + 1, "Right Title")
    # Insert Right Company Name after Company Name for Emails
    new_header.insert(company_idx + 1, "Right Company Name")

    # Adjust title_idx after company insertion if title comes after company
    if title_idx > company_idx:
        title_idx += 1  # shifted by Right Company Name insertion

    log.append(f"Loading few-shot training examples…")
    title_text, company_text = _load_examples()
    log.append(
        f"Loaded {title_text.count(chr(10)) + 1 if title_text else 0} title examples, "
        f"{company_text.count(chr(10)) + 1 if company_text else 0} company examples."
    )

    system_prompt = _build_system_prompt(title_text, company_text)
    client = anthropic.Anthropic(api_key=api_key)

    total = len(data_rows)
    processed = 0
    errors = 0

    # Prepare output rows (copy of data_rows with 2 extra columns)
    out_rows: list[list[str]] = []
    for row in data_rows:
        new_row = list(row)
        # Insert Right Title placeholder after Title
        new_row.insert(title_idx + 1, "")
        # Insert Right Company Name placeholder after Company Name for Emails
        new_row.insert(company_idx + 1, "")
        out_rows.append(new_row)

    # Process in batches
    batches = [
        list(enumerate(data_rows[i : i + BATCH_SIZE], start=i))
        for i in range(0, total, BATCH_SIZE)
    ]

    for batch_num, batch in enumerate(batches):
        if should_stop and should_stop():
            log.append(f"Stopped by user after {processed} rows.")
            break

        if on_progress:
            pct = processed / total if total else 0
            on_progress(pct, f"Batch {batch_num + 1}/{len(batches)} ({processed}/{total} rows)…")

        user_msg = _build_user_message(batch)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            results: list[dict] = parsed.get("results", [])

            for item in results:
                local_i = item.get("i")
                if local_i is None or local_i >= len(batch):
                    continue
                global_i = batch[local_i][0]
                out_row = out_rows[global_i]

                right_company = item.get("right_company", "").strip()
                right_title = item.get("right_title", "").strip()
                ai_country = item.get("country", "").strip()
                ai_state = item.get("state", "").strip()
                ai_city = item.get("city", "").strip()

                out_row[company_idx + 1] = right_company

                # title_idx already shifted by 1 for Right Company Name insertion
                out_row[title_idx + 1] = right_title

                # Location: only fill empty fields
                # country/state/city indices in out_row (accounting for 2 inserted cols)
                # Original COL_COUNTRY=13, COL_STATE=14, COL_CITY=15 → shifted by 2
                country_i = COL_COUNTRY + 2
                state_i = COL_STATE + 2
                city_i = COL_CITY + 2

                if not out_row[country_i]:
                    out_row[country_i] = ai_country
                if not out_row[state_i]:
                    out_row[state_i] = ai_state
                if not out_row[city_i]:
                    out_row[city_i] = ai_city

            processed += len(batch)

        except json.JSONDecodeError as e:
            errors += 1
            log.append(f"Batch {batch_num + 1}: JSON parse error — {e}")
            processed += len(batch)
        except anthropic.APIError as e:
            errors += 1
            log.append(f"Batch {batch_num + 1}: API error — {e}")
            processed += len(batch)

    if on_progress:
        on_progress(1.0, f"Done. {processed} rows processed, {errors} batch errors.")

    log.append(f"Step 2 complete. {processed}/{total} rows, {errors} batch errors.")
    return [new_header] + out_rows, log
