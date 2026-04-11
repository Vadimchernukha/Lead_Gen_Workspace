"""Apollo Cleaner tool — rule-based cleaning + AI enrichment + email domain check."""
from __future__ import annotations

import os
import re
import threading
from io import BytesIO

import pandas as pd
import streamlit as st

import core.apollo_pipeline as pipeline
import core.apollo_ai_step as ai_step
import core.apollo_email_gate as email_gate


def _get_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.strip()


def run() -> None:
    st.title("Apollo Cleaner")
    st.caption(
        "Upload an Apollo.io CSV export → rule-based cleaning → AI enrichment (Claude) → email domain check."
    )

    # ── settings row ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline settings", expanded=True):
        s_col1, s_col2 = st.columns([3, 2])
        with s_col1:
            run_step1 = st.checkbox("Step 1 — Rule-based cleaning", value=True)
            run_step2 = st.checkbox("Step 2 — AI enrichment (Claude Sonnet)", value=False)
            run_step3 = st.checkbox("Step 3 — Email domain check", value=True)
        with s_col2:
            api_key_input = st.text_input(
                "Claude API Key (override)",
                type="password",
                placeholder="sk-ant-… (leave blank to use secrets.toml)",
                help="Required only for Step 2. Falls back to ANTHROPIC_API_KEY in secrets.",
            )
            if run_step2 and not api_key_input and not _get_api_key():
                st.warning("⚠️ Claude API key not found — required for Step 2.")

    api_key = api_key_input.strip() if api_key_input.strip() else _get_api_key()

    st.divider()

    # ── file upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Apollo CSV export",
        type=["csv"],
        help="Accepts UTF-8 or UTF-8-BOM, any delimiter (comma / semicolon / tab).",
    )

    if not uploaded:
        return

    raw_bytes = uploaded.read()

    try:
        peek_df = pd.read_csv(BytesIO(raw_bytes), encoding="utf-8-sig", sep=None, engine="python", nrows=5)
    except Exception:
        peek_df = pd.DataFrame()

    try:
        import csv, io as _io
        text = raw_bytes.decode("utf-8-sig")
        full_df_len = max(0, sum(1 for _ in csv.reader(_io.StringIO(text))) - 1)
    except Exception:
        full_df_len = 0

    c1, c2 = st.columns(2)
    c1.metric("Rows", full_df_len)
    c2.metric("Columns", len(peek_df.columns) if not peek_df.empty else "—")

    with st.expander("Preview (first 5 rows)", expanded=False):
        if not peek_df.empty:
            st.dataframe(peek_df, use_container_width=True)
        else:
            st.warning("Could not preview file.")

    st.divider()

    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if not run_btn:
        return

    if not (run_step1 or run_step2 or run_step3):
        st.error("Select at least one pipeline step.")
        return

    if run_step2 and not api_key:
        st.error("Claude API key required for Step 2.")
        return

    stop_event = threading.Event()
    all_logs: list[str] = []
    current_rows: list[list[str]] = []

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if run_step1:
        with st.status("Step 1 — Rule-based cleaning…", expanded=True) as s1:
            prog1 = st.progress(0.0)
            try:
                rows1, log1 = pipeline.run(raw_bytes)
                all_logs.extend(log1)
                current_rows = rows1
                prog1.progress(1.0)
                s1.update(label=f"✅ Step 1 — {len(rows1) - 1} rows", state="complete", expanded=False)
            except Exception as exc:
                s1.update(label=f"❌ Step 1 failed: {exc}", state="error")
                all_logs.append(f"Step 1 ERROR: {exc}")
                return
    else:
        try:
            raw_header, raw_data = pipeline.parse_csv_bytes(raw_bytes)
            current_rows = [raw_header] + raw_data
            all_logs.append("Step 1 skipped.")
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
            return

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if run_step2:
        with st.status("Step 2 — AI enrichment…", expanded=True) as s2:
            prog2 = st.progress(0.0)
            msg2 = st.empty()

            def _prog2(pct: float, text: str) -> None:
                prog2.progress(min(pct, 1.0))
                msg2.caption(text)

            try:
                rows2, log2 = ai_step.run(
                    current_rows,
                    api_key=api_key,
                    on_progress=_prog2,
                    should_stop=stop_event.is_set,
                )
                all_logs.extend(log2)
                current_rows = rows2
                prog2.progress(1.0)
                s2.update(label=f"✅ Step 2 — {len(rows2) - 1} rows", state="complete", expanded=False)
            except Exception as exc:
                s2.update(label=f"❌ Step 2 failed: {exc}", state="error")
                all_logs.append(f"Step 2 ERROR: {exc}")
                return

    # ── Step 3 ────────────────────────────────────────────────────────────────
    domain_counts: dict[str, int] = {}
    if run_step3:
        with st.status("Step 3 — Email domain checks…", expanded=True) as s3:
            prog3 = st.progress(0.0)
            msg3 = st.empty()

            def _prog3(pct: float, text: str) -> None:
                prog3.progress(min(pct, 1.0))
                msg3.caption(text)

            try:
                rows3, log3 = email_gate.run(
                    current_rows,
                    on_progress=_prog3,
                    should_stop=stop_event.is_set,
                )
                all_logs.extend(log3)
                current_rows = rows3
                prog3.progress(1.0)
                for entry in log3:
                    if "Step 3 complete" in entry:
                        for label in ("Match", "No Match", "Dead"):
                            m = re.search(rf"{label}: (\d+)", entry)
                            if m:
                                domain_counts[label] = int(m.group(1))
                s3.update(label=f"✅ Step 3 — {len(rows3) - 1} rows", state="complete", expanded=False)
            except Exception as exc:
                s3.update(label=f"❌ Step 3 failed: {exc}", state="error")
                all_logs.append(f"Step 3 ERROR: {exc}")
                return

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Results")

    result_header = current_rows[0] if current_rows else []
    result_data = current_rows[1:] if len(current_rows) > 1 else []

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Rows", len(result_data))
    r2.metric("Match",    domain_counts.get("Match", "—"))
    r3.metric("No Match", domain_counts.get("No Match", "—"))
    r4.metric("Dead",     domain_counts.get("Dead", "—"))

    if result_data:
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(pd.DataFrame(result_data[:10], columns=result_header), use_container_width=True)

        csv_bytes = pipeline.rows_to_csv_bytes(result_header, result_data)
        download_name = uploaded.name.removesuffix(".csv") + "_cleaned.csv"
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv_bytes,
            file_name=download_name,
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

    with st.expander("Full Processing Log", expanded=False):
        st.code("\n".join(all_logs), language=None)
