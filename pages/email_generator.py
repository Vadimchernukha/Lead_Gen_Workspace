"""Email Generator — Streamlit page."""
from __future__ import annotations

import smtplib
import socket

import dns.resolver
import pandas as pd
import streamlit as st
from unidecode import unidecode

MAX_FORMATS = 9
SMTP_TIMEOUT = 10
EHLO_HOSTNAME = "google.com"


# ---------------------------------------------------------------------------
# Email pattern generation
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    if not value or isinstance(value, float):
        return ""
    return unidecode(str(value).strip().lower())


def generate_emails(first_name, last_name, domain) -> list[str]:
    f = _clean(first_name)
    l = _clean(last_name)
    d = _clean(domain)

    if not d:
        return []

    if f and l:
        emails = [
            f"{f}.{l}@{d}",
            f"{f[0]}{l}@{d}",
            f"{f}@{d}",
            f"{f[0]}.{l}@{d}",
            f"{f}{l}@{d}",
            f"{f}{l[0]}@{d}",
            f"{f}.{l[0]}@{d}",
            f"{l}@{d}",
            f"{f[0]}{l[0]}@{d}",
        ]
    elif f:
        emails = [f"{f}@{d}"]
    elif l:
        emails = [f"{l}@{d}"]
    else:
        return []

    return emails[:MAX_FORMATS]


# ---------------------------------------------------------------------------
# DNS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_mx_records(domain: str) -> list[str]:
    try:
        records = dns.resolver.resolve(domain, "MX")
        mx_hosts = sorted(records, key=lambda r: r.preference)
        return [str(r.exchange).rstrip(".") for r in mx_hosts]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SMTP — with STARTTLS + disguised EHLO
# ---------------------------------------------------------------------------

def smtp_check(email: str, mx_host: str, from_address: str = "verify@example.com") -> str:
    try:
        with smtplib.SMTP(timeout=SMTP_TIMEOUT) as smtp:
            smtp.connect(mx_host, 25)
            smtp.ehlo(EHLO_HOSTNAME)
            if smtp.has_extn("STARTTLS"):
                smtp.starttls()
                smtp.ehlo(EHLO_HOSTNAME)
            smtp.mail(from_address)
            code, _ = smtp.rcpt(email)
            if code == 250:
                return "valid"
            elif code in (550, 551, 553):
                return "invalid"
            else:
                return "unverified"
    except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, socket.timeout, OSError):
        return "unverified"
    except Exception:
        return "unverified"


@st.cache_data(show_spinner=False)
def is_catch_all(domain: str, mx_host: str) -> bool:
    fake = f"zz_no_such_user_xkq7@{domain}"
    return smtp_check(fake, mx_host) == "valid"


# ---------------------------------------------------------------------------
# Verification with pattern caching
# ---------------------------------------------------------------------------

def verify_row(
    emails: list[str],
    domain: str,
    pattern_cache: dict[str, int],
) -> tuple[str, str]:
    """
    Returns (best_email, status).

    pattern_cache maps domain → index of the confirmed working pattern.
    On a cache hit the function returns instantly without any SMTP call.
    """
    if not emails:
        return "", "unverified"

    mx_list = get_mx_records(domain)
    if not mx_list:
        return "", "no-mx"

    mx = mx_list[0]

    if is_catch_all(domain, mx):
        # Pick cached pattern if available, otherwise pattern 0
        idx = pattern_cache.get(domain, 0)
        idx = min(idx, len(emails) - 1)
        return emails[idx], "catch-all"

    # Cache hit — skip SMTP entirely
    if domain in pattern_cache:
        idx = min(pattern_cache[domain], len(emails) - 1)
        return emails[idx], "valid"

    # Full probe
    last_status = "unverified"
    for i, email in enumerate(emails):
        status = smtp_check(email, mx)
        if status == "valid":
            pattern_cache[domain] = i
            return email, "valid"
        elif status == "invalid":
            last_status = "invalid"

    if last_status == "invalid":
        return emails[0], "blocked"

    return "", last_status


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def run() -> None:
    st.title("Email Generator 📧")
    st.caption(
        "Upload a CSV with **FirstName**, **LastName**, **Domain** columns — "
        "the tool generates all probable email patterns, probes each domain via SMTP, "
        "and returns one verified address per contact."
    )

    with st.expander("Status legend", expanded=False):
        st.markdown(
            "- **valid** — confirmed by server  \n"
            "- **catch-all** — server accepts everything, best-guess shown  \n"
            "- **blocked** — server rejects all probes (anti-spam), best-guess shown  \n"
            "- **no-mx** — domain has no MX records  \n"
            "- **unverified** — connection failed / timeout  \n"
            "- **skipped** — contact already had an email in the source file"
        )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    required_cols = {"FirstName", "LastName", "Domain"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(sorted(missing))}")
        return

    st.info(f"Loaded **{len(df)}** rows.")

    if not st.button("Generate & Verify", type="primary"):
        return

    # Per-session pattern cache persists across re-runs (same session)
    if "pattern_cache" not in st.session_state:
        st.session_state.pattern_cache = {}
    pattern_cache: dict[str, int] = st.session_state.pattern_cache

    total = len(df)
    real_emails: list[str] = []
    statuses: list[str] = []

    progress = st.progress(0, text="Starting…")

    for idx, row in df.iterrows():
        row_num = int(str(idx)) + 1  # type: ignore[arg-type]

        existing = row.get("Email", "") if "Email" in df.columns else ""
        if pd.notna(existing) and str(existing).strip():
            real_emails.append(str(existing).strip())
            statuses.append("skipped")
            progress.progress(row_num / total, text=f"Row {row_num}/{total} — skipped")
            continue

        domain = str(row["Domain"]).strip() if pd.notna(row["Domain"]) else ""
        emails = generate_emails(row["FirstName"], row["LastName"], domain)

        if not emails or not domain:
            real_emails.append("")
            statuses.append("unverified")
            progress.progress(row_num / total, text=f"Row {row_num}/{total} — no domain")
            continue

        cached = domain in pattern_cache
        label = f"Row {row_num}/{total} — {'cached ⚡' if cached else f'probing {domain}…'}"
        progress.progress(row_num / total, text=label)

        email, status = verify_row(emails, domain, pattern_cache)
        real_emails.append(email)
        statuses.append(status)

    progress.empty()

    # Build output — keep only source columns + two new ones
    out = df.copy()
    out["Real_Email"] = real_emails
    out["Status"] = statuses

    st.success(f"Done — {total} rows processed, {len(pattern_cache)} domain patterns cached.")

    st.subheader("Verification stats")
    stats = (
        pd.Series(statuses, name="Count")
        .value_counts()
        .rename_axis("Status")
        .reset_index()
    )
    st.dataframe(stats, use_container_width=True)

    st.subheader("Preview (first 10 rows)")
    st.dataframe(out[["FirstName", "LastName", "Domain", "Real_Email", "Status"]].head(10),
                 use_container_width=True)

    csv_buffer = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download result (CSV)",
        data=csv_buffer,
        file_name="emails_verified.csv",
        mime="text/csv",
    )
