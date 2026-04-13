"""Apollo Cleaner — Streamlit page."""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from core.apollo_cleaner import run as cleaner_run

st.set_page_config(page_title="Apollo Cleaner", layout="wide")

st.title("Apollo Cleaner")
st.caption(
    "Upload an Apollo.io CSV export → clean names & deduplicate → "
    "URL/industry normalisation → AI enrichment (Claude) → download."
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Leave empty to skip the LLM enrichment step.",
    )

    st.divider()
    st.subheader("Pipeline steps")
    step_llm = st.checkbox("AI enrichment (Claude)", value=bool(api_key))

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("Upload Apollo CSV export", type=["csv"])

if uploaded:
    raw_bytes = uploaded.read()
    try:
        preview_df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, nrows=5)
        with st.expander("Preview (first 5 rows)", expanded=False):
            st.dataframe(preview_df, width="stretch")
    except Exception:
        st.warning("Could not preview file.")

    st.divider()
    run_btn = st.button("▶ Run Pipeline", type="primary", width="stretch")

    if run_btn:
        effective_key = api_key.strip() if step_llm else None

        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        log_expander = st.expander("Processing log", expanded=False)

        def on_progress(frac: float, msg: str) -> None:
            progress_bar.progress(min(frac, 1.0))
            status_text.info(msg)

        with st.spinner("Running pipeline…"):
            try:
                final_df, log_lines = cleaner_run(
                    raw_bytes,
                    api_key=effective_key,
                    on_progress=on_progress,
                )
                with log_expander:
                    for line in log_lines:
                        st.text(line)

                st.success(f"Done! {len(final_df)} rows × {len(final_df.columns)} columns.")

                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Output rows", len(final_df))
                c2.metric("Columns", len(final_df.columns))
                domain_matches = int(final_df["Email_Domain_Match"].astype(str).str.lower().eq("true").sum())
                c3.metric("Email/domain match", domain_matches)

                # Preview
                st.subheader("Preview (first 10 rows)")
                st.dataframe(final_df.head(10), width="stretch")

                # Download
                csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇ Download cleaned CSV",
                    data=csv_bytes,
                    file_name="cleaned_apollo_contacts.csv",
                    mime="text/csv",
                    width="stretch",
                )

            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                raise
