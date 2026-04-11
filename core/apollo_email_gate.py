"""Step 3 — Email domain check via DNS + HTTP."""
from __future__ import annotations

import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import httpx

WORKERS = 10
HTTP_TIMEOUT = 10.0

STATUS_MATCH = "Match"
STATUS_NO_MATCH = "No Match"
STATUS_DEAD = "Dead"


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def extract_email_domain(email: str) -> str:
    email = email.strip()
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().casefold()


def normalise_domain(raw: str) -> str:
    """Strip scheme, www., trailing slashes, paths → bare domain."""
    raw = raw.strip()
    if not raw:
        return ""
    raw = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^www\.", "", raw, flags=re.IGNORECASE)
    # Drop path, query, fragment
    raw = raw.split("/")[0].split("?")[0].split("#")[0]
    return raw.casefold()


def _dns_ok(domain: str) -> bool:
    try:
        socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
        return True
    except socket.gaierror:
        return False


def _http_alive(domain: str) -> bool:
    """Try http then https; follow redirects; return True if any succeeds."""
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT) as client:
                resp = client.get(url)
                if resp.status_code < 600:
                    return True
        except Exception:
            continue
    return False


def check_domain(email_domain: str, website_domain: str) -> str:
    """Return Match / No Match / Dead for a single row."""
    if not email_domain:
        return ""

    if email_domain == website_domain:
        return STATUS_MATCH

    # Need HTTP check
    if not _dns_ok(email_domain):
        return STATUS_DEAD

    if not _http_alive(email_domain):
        return STATUS_DEAD

    return STATUS_NO_MATCH


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def _find_col(header: list[str], name: str) -> int | None:
    norm = lambda s: " ".join(s.strip().split()).casefold()
    target = norm(name)
    for i, h in enumerate(header):
        if norm(h) == target:
            return i
    return None


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------

def run(
    rows: list[list[str]],
    on_progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[list[str]], list[str]]:
    """
    Run Step 3: add Email Domain and Domain Check columns.

    `rows[0]` must be the header row.
    Returns (rows, log).
    """
    log: list[str] = []

    if not rows:
        return rows, ["ERROR: no rows passed to email_gate"]

    header = list(rows[0])
    data_rows = [list(r) for r in rows[1:]]

    if not data_rows:
        return rows, ["No data rows to process."]

    email_idx = _find_col(header, "Email")
    website_idx = _find_col(header, "Website")

    if email_idx is None:
        log.append("WARNING: Email column not found.")
    if website_idx is None:
        log.append("WARNING: Website column not found.")

    # Insert new columns after Email
    insert_after = email_idx if email_idx is not None else len(header) - 1
    new_header = list(header)
    new_header.insert(insert_after + 1, "Email Domain")
    new_header.insert(insert_after + 2, "Domain Check")

    # Adjust website_idx if it falls after the insertion point
    if website_idx is not None and website_idx > insert_after:
        website_idx += 2

    email_domain_idx = insert_after + 1
    domain_check_idx = insert_after + 2

    # Extend each data row with 2 placeholder slots
    out_rows: list[list[str]] = []
    for row in data_rows:
        new_row = list(row)
        new_row.insert(insert_after + 1, "")  # Email Domain
        new_row.insert(insert_after + 2, "")  # Domain Check
        out_rows.append(new_row)

    # Pre-compute domains
    pairs: list[tuple[int, str, str]] = []  # (row_index, email_domain, website_domain)
    for i, row in enumerate(out_rows):
        email = row[email_idx] if email_idx is not None and email_idx < len(row) else ""
        website = row[website_idx] if website_idx is not None and website_idx < len(row) else ""
        ed = extract_email_domain(email)
        wd = normalise_domain(website)
        row[email_domain_idx] = ed
        if ed:
            pairs.append((i, ed, wd))

    total = len(pairs)
    done = 0

    log.append(f"Checking {total} rows with valid email domains…")

    if on_progress:
        on_progress(0.0, f"Starting domain checks for {total} rows…")

    def _task(item: tuple[int, str, str]) -> tuple[int, str]:
        idx, ed, wd = item
        if should_stop and should_stop():
            return idx, ""
        status = check_domain(ed, wd)
        return idx, status

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_task, p): p[0] for p in pairs}
        for future in as_completed(futures):
            if should_stop and should_stop():
                executor.shutdown(wait=False, cancel_futures=True)
                log.append("Stopped by user.")
                break
            idx, status = future.result()
            out_rows[idx][domain_check_idx] = status
            done += 1
            if on_progress and total:
                on_progress(done / total, f"Domain checks: {done}/{total}…")

    # Summary
    counts: dict[str, int] = {STATUS_MATCH: 0, STATUS_NO_MATCH: 0, STATUS_DEAD: 0, "": 0}
    for row in out_rows:
        s = row[domain_check_idx]
        counts[s] = counts.get(s, 0) + 1

    log.append(
        f"Step 3 complete. Match: {counts[STATUS_MATCH]}, "
        f"No Match: {counts[STATUS_NO_MATCH]}, "
        f"Dead: {counts[STATUS_DEAD]}, "
        f"No email: {counts.get('', 0)}."
    )

    if on_progress:
        on_progress(1.0, "Domain checks done.")

    return [new_header] + out_rows, log
