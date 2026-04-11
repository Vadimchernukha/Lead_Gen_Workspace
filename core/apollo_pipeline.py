"""Step 1 — rule-based cleaning, no external APIs."""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from core.industry_map import INDUSTRY_MAPPING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "Company Name for Emails",
    "Website",
    "Industry",
    "Employee Range",
    "Company Country",
    "Company State",
    "Company City",
    "First Name",
    "Last Name",
    "Title",
    "Email",
    "Person Linkedin Url",
    "Company Linkedin Url",
    "Country",
    "State",
    "City",
    "Apollo Contact Id",
    "Apollo Account Id",
]

EMPLOYEE_BANDS = [
    (1, 10, "1-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1000"),
    (1001, 5000, "1001-5000"),
    (5001, 10000, "5001-10000"),
    (10001, float("inf"), "10001+"),
]

KNOWN_BAND_STRINGS = {b for _, _, b in EMPLOYEE_BANDS}

# Aliases: normalised key → canonical output column name
COLUMN_ALIASES: dict[str, str] = {
    # Employee count
    "employees": "# employees",
    "number of employees": "# employees",
    "# employees": "# employees",
    "num employees": "# employees",
    # Apollo IDs
    "apollo contact id": "Apollo Contact Id",
    "apollo contact id ": "Apollo Contact Id",
    "apollocontactid": "Apollo Contact Id",
    "apollo account id": "Apollo Account Id",
    "apollo account id ": "Apollo Account Id",
    "apolloaccountid": "Apollo Account Id",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_header(s: str) -> str:
    """Normalise a header: strip, collapse whitespace, casefold."""
    return " ".join(s.strip().split()).casefold()


def strip_url_scheme(url: str) -> str:
    """Remove http://, https://, www. prefix."""
    url = url.strip()
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    url = re.sub(r"^www\.", "", url, flags=re.IGNORECASE)
    return url.rstrip("/")


def to_employee_band(raw: str) -> str:
    """Convert a raw employee count string to a band label."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw in KNOWN_BAND_STRINGS:
        return raw
    # Already a band-like string with dash?
    if re.match(r"^\d[\d,]*\s*[-–]\s*\d[\d,]*\+?$", raw):
        return raw
    if raw.endswith("+") and re.match(r"^\d[\d,]+\+$", raw):
        return raw
    # Parse numeric value
    numeric_str = re.sub(r"[,\s]", "", raw)
    try:
        n = int(numeric_str)
    except ValueError:
        return raw  # can't parse — return as-is
    for lo, hi, label in EMPLOYEE_BANDS:
        if lo <= n <= hi:
            return label
    return raw


def parse_csv_bytes(data: bytes) -> tuple[list[str], list[list[str]]]:
    """Parse CSV bytes (UTF-8-BOM, auto-detect delimiter)."""
    text = data.decode("utf-8-sig")
    # Auto-detect delimiter
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def rows_to_csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    """Serialise to UTF-8-BOM CSV with LF line endings."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------

def run(data: bytes) -> tuple[list[list[str]], list[str]]:
    """
    Run Step 1 rule-based cleaning.

    Returns (rows, log) where rows[0] is the header row.
    """
    log: list[str] = []
    raw_header, raw_rows = parse_csv_bytes(data)

    if not raw_header:
        return [], ["ERROR: empty or unreadable CSV"]

    log.append(f"Parsed {len(raw_rows)} data rows, {len(raw_header)} columns.")

    # Build normalised header → original index map
    norm_to_idx: dict[str, int] = {}
    for i, h in enumerate(raw_header):
        key = norm_header(h)
        # Apply alias resolution
        key = COLUMN_ALIASES.get(key, key)
        if key not in norm_to_idx:
            norm_to_idx[key] = i

    def get_col(row: list[str], canonical_name: str, default: str = "") -> str:
        key = norm_header(canonical_name)
        key = COLUMN_ALIASES.get(key, key)
        idx = norm_to_idx.get(key)
        if idx is None:
            return default
        return row[idx] if idx < len(row) else default

    # Detect employee column (could be "# Employees", "Employees", etc.)
    employee_col_key: str | None = None
    for alias_key, resolved in COLUMN_ALIASES.items():
        if resolved == "# employees" and alias_key in norm_to_idx:
            employee_col_key = alias_key
            break
    if employee_col_key is None and "# employees" in norm_to_idx:
        employee_col_key = "# employees"

    if employee_col_key:
        log.append(f"Employee column found at index {norm_to_idx[employee_col_key]}.")
    else:
        log.append("WARNING: No employee count column found.")

    # Detect Apollo ID columns
    contact_id_key = next(
        (k for k, v in COLUMN_ALIASES.items() if v == "Apollo Contact Id" and k in norm_to_idx),
        "apollo contact id" if "apollo contact id" in norm_to_idx else None,
    )
    account_id_key = next(
        (k for k, v in COLUMN_ALIASES.items() if v == "Apollo Account Id" and k in norm_to_idx),
        "apollo account id" if "apollo account id" in norm_to_idx else None,
    )

    url_fields = {"website", "person linkedin url", "company linkedin url"}
    missing_cols: set[str] = set()

    output_rows: list[list[str]] = []

    for row in raw_rows:
        out: dict[str, str] = {}

        for col in OUTPUT_COLUMNS:
            col_key = norm_header(col)

            # Special handling
            if col == "Employee Range":
                raw_val = ""
                if employee_col_key:
                    idx = norm_to_idx.get(employee_col_key)
                    raw_val = (row[idx] if idx is not None and idx < len(row) else "").strip()
                out["Employee Range"] = to_employee_band(raw_val)
                continue

            if col == "Apollo Contact Id":
                idx = norm_to_idx.get(contact_id_key) if contact_id_key else None
                out["Apollo Contact Id"] = (row[idx] if idx is not None and idx < len(row) else "").strip()
                continue

            if col == "Apollo Account Id":
                idx = norm_to_idx.get(account_id_key) if account_id_key else None
                out["Apollo Account Id"] = (row[idx] if idx is not None and idx < len(row) else "").strip()
                continue

            # General lookup
            idx = norm_to_idx.get(col_key)
            if idx is None:
                if col not in missing_cols:
                    missing_cols.add(col)
                val = ""
            else:
                val = row[idx] if idx < len(row) else ""

            # URL stripping
            if col_key in url_fields:
                val = strip_url_scheme(val)

            # Industry normalisation
            if col == "Industry":
                mapped = INDUSTRY_MAPPING.get(val.strip().casefold())
                if mapped:
                    val = mapped

            out[col] = val.strip()

        # Fill person location from company if countries match and person field empty
        comp_country = out.get("Company Country", "").strip().casefold()
        pers_country = out.get("Country", "").strip().casefold()

        if comp_country and (not pers_country or pers_country == comp_country):
            if not out.get("Country"):
                out["Country"] = out.get("Company Country", "")
            if not out.get("State"):
                out["State"] = out.get("Company State", "")
            if not out.get("City"):
                out["City"] = out.get("Company City", "")

        output_rows.append([out[c] for c in OUTPUT_COLUMNS])

    if missing_cols:
        log.append(f"Columns not found (set to empty): {', '.join(sorted(missing_cols))}")

    log.append(f"Step 1 complete. Output: {len(output_rows)} rows × {len(OUTPUT_COLUMNS)} columns.")
    return [OUTPUT_COLUMNS] + output_rows, log
