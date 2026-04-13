"""Email Generator — Streamlit page."""
from __future__ import annotations

import smtplib
import socket

import dns.resolver
import pandas as pd
import streamlit as st
from unidecode import unidecode

MAX_FORMATS = 8
SMTP_TIMEOUT = 10


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
        ]
    elif f:
        emails = [f"{f}@{d}"]
    elif l:
        emails = [f"{l}@{d}"]
    else:
        return []

    return emails[:MAX_FORMATS]


@st.cache_data(show_spinner=False)
def get_mx_records(domain: str) -> list[str]:
    try:
        records = dns.resolver.resolve(domain, "MX")
        mx_hosts = sorted(records, key=lambda r: r.preference)
        return [str(r.exchange).rstrip(".") for r in mx_hosts]
    except Exception:
        return []


def smtp_check(email: str, mx_host: str, from_address: str = "verify@example.com") -> str:
    try:
        with smtplib.SMTP(timeout=SMTP_TIMEOUT) as smtp:
            smtp.connect(mx_host, 25)
            smtp.ehlo_or_helo_if_needed()
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


def verify_emails_for_row(emails: list[str], domain: str) -> tuple[str, str]:
    if not emails:
        return ("", "unverified")

    mx_list = get_mx_records(domain)
    if not mx_list:
        return ("", "no-mx")

    mx = mx_list[0]

    if is_catch_all(domain, mx):
        return (emails[0], "catch-all")

    last_status = "unverified"
    for email in emails:
        status = smtp_check(email, mx)
        if status == "valid":
            return (email, "valid")
        elif status == "invalid":
            last_status = "invalid"

    return ("", last_status)


def run() -> None:
    st.title("Email Generator 📧")
    st.caption(
        "Upload a CSV with **FirstName**, **LastName**, **Domain** columns → "
        "generate probable corporate email formats → optional SMTP verification → download."
    )

    verify_mode = st.checkbox(
        "Run SMTP verification (slower, but filters out invalid addresses)",
        value=True,
    )

    if verify_mode:
        st.info(
            "Verification statuses: **valid** — confirmed | "
            "**invalid** — rejected by server | "
            "**catch-all** — server accepts everything | "
            "**no-mx** — domain has no MX records | "
            "**unverified** — server blocked the check"
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
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.info(f"Loaded rows: **{len(df)}**")

    if not st.button("Generate emails", type="primary"):
        return

    for i in range(1, MAX_FORMATS + 1):
        col_name = f"Generated_Email_{i}"
        if col_name not in df.columns:
            df[col_name] = ""

    if verify_mode:
        df["Verified_Email"] = ""
        df["Email_Status"] = ""

    progress = st.progress(0, text="Processing rows…")
    total = len(df)

    for idx, row in df.iterrows():
        if "Email" in df.columns:
            existing = row.get("Email", "")
            if pd.notna(existing) and str(existing).strip():
                progress.progress(
                    (idx + 1) / total,
                    text=f"Row {idx + 1} of {total} — skipped (email exists)",
                )
                continue

        domain = str(row["Domain"]).strip() if pd.notna(row["Domain"]) else ""
        emails = generate_emails(row["FirstName"], row["LastName"], domain)

        for i, email in enumerate(emails, start=1):
            df.at[idx, f"Generated_Email_{i}"] = email

        if verify_mode and emails and domain:
            progress.progress(
                (idx + 1) / total,
                text=f"Row {idx + 1} of {total} — verifying {domain}…",
            )
            best_email, status = verify_emails_for_row(emails, domain)
            df.at[idx, "Verified_Email"] = best_email
            df.at[idx, "Email_Status"] = status
        else:
            progress.progress((idx + 1) / total, text=f"Row {idx + 1} of {total}")

    progress.empty()
    st.success(f"Done! Processed rows: {total}")

    if verify_mode and "Email_Status" in df.columns:
        st.subheader("Verification stats")
        stats = df["Email_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        st.dataframe(stats, use_container_width=True)

    st.subheader("Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    csv_buffer = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download result (CSV)",
        data=csv_buffer,
        file_name="emails_generated.csv",
        mime="text/csv",
    )
