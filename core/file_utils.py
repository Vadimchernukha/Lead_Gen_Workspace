import io
import pandas as pd
import streamlit as st


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile as a DataFrame, handling common encoding issues."""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def download_button_csv(df: pd.DataFrame, filename: str) -> None:
    """Render a Streamlit download button that exports df as a CSV file."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download {filename}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
