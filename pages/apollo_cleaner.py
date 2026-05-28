"""Apollo Cleaner — Streamlit page."""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from core.apollo_cleaner import run as cleaner_run


def run() -> None:
    st.title("Apollo Cleaner")
    st.caption(
        "Upload an Apollo.io CSV export → clean names & deduplicate → "
        "URL/industry normalisation → AI enrichment (Claude) → download."
    )

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    with st.expander("⚙️ Settings", expanded=True):
        api_key  = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Leave empty to skip the LLM enrichment step.",
        )
        step_llm = st.checkbox("AI enrichment (Claude)", value=bool(api_key))

    # -------------------------------------------------------------------------
    # File upload
    # -------------------------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload Apollo CSV export(s)",
        type=["csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        dfs = []
        bad = []
        for f in uploaded_files:
            try:
                dfs.append(pd.read_csv(io.BytesIO(f.read()), dtype=str, low_memory=False))
            except Exception:
                bad.append(f.name)

        if bad:
            st.warning(f"Could not read: {', '.join(bad)}")

        if not dfs:
            st.stop()

        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        if len(uploaded_files) > 1:
            st.info(f"{len(uploaded_files)} files merged → {len(combined_df)} total rows before deduplication.")

        raw_bytes = combined_df.to_csv(index=False).encode("utf-8-sig")

        with st.expander("Preview (first 5 rows)", expanded=False):
            st.dataframe(combined_df.head(5), use_container_width=True)

        st.divider()
        run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

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

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Output rows", len(final_df))
                    c2.metric("Columns", len(final_df.columns))
                    domain_matches = int(final_df["Email_Domain_Match"].astype(str).str.lower().eq("true").sum())
                    c3.metric("Email/domain match", domain_matches)

                    st.subheader("Preview (first 10 rows)")
                    st.dataframe(final_df.head(10), use_container_width=True)

                    csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇ Download cleaned CSV",
                        data=csv_bytes,
                        file_name="cleaned_apollo_contacts.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
                    raise
