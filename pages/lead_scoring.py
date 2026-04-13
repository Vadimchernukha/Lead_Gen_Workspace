"""
Lead Scoring tool — Apollo export + ICP profiles + Claude Haiku 4.5 waterfall.
Scoring runs in a background thread; UI polls and reruns until done.
"""
from __future__ import annotations

import datetime
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from core.scoring_logic import (
    DEFAULT_COST_INPUT_PER_MILLION_USD,
    DEFAULT_COST_OUTPUT_PER_MILLION_USD,
    estimate_cost_usd,
    export_colored_xlsx,
    score_company_row,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROFILES_PATH = BASE_DIR / "profiles.yaml"
RESULTS_DIR = BASE_DIR / "results"


# ── persistent job store ──────────────────────────────────────────────────────

@st.cache_resource
def _get_jobs() -> dict[str, dict[str, Any]]:
    return {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    if not PROFILES_PATH.exists():
        return {}
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.strip()


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "ls_uploaded_df": None,
        "ls_upload_name": "",
        "ls_icp_profile": None,
        "ls_job_id": None,
        "ls_cost_input": float(DEFAULT_COST_INPUT_PER_MILLION_USD),
        "ls_cost_output": float(DEFAULT_COST_OUTPUT_PER_MILLION_USD),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── background worker ─────────────────────────────────────────────────────────

def _worker(job_id: str, api_key: str, icp_desc: str, df: pd.DataFrame) -> None:
    job = _get_jobs()[job_id]
    stop_event: threading.Event = job["stop_event"]
    out_rows: list[dict] = []
    total_in = total_out = 0
    errors = 0

    for idx, row in df.iterrows():
        if stop_event.is_set():
            job["log"].append("⛔ Остановлено пользователем.")
            break

        row_dict = row.to_dict()
        company = row_dict.get("Company Name", row_dict.get("Company", f"row_{idx}"))
        try:
            cname = str(company) if company is not None and str(company) != "nan" else f"row_{idx}"
        except Exception:
            cname = f"row_{idx}"

        job["processed"] = len(out_rows)

        try:
            result, (in_tok, out_tok) = score_company_row(api_key, icp_desc, row_dict)
            total_in += in_tok
            total_out += out_tok
        except Exception as e:
            errors += 1
            result = {
                "ICP_Status": "MANUAL_REVIEW",
                "Reason": f"Unhandled: {e}",
                "Data_Source": "Error",
            }

        status = result.get("ICP_Status", "?")
        source = result.get("Data_Source", "")
        reason = (result.get("Reason") or "")[:70]
        icon = {"YES": "✅", "NO": "❌", "MAYBE": "⚠️"}.get(status, "🔵")
        job["log"].append(f"{icon} {cname}  [{source}]  {reason}")

        out_rows.append({**row_dict, **result})
        job["processed"] = len(out_rows)
        job["errors"] = errors
        job["input_tokens"] = total_in
        job["output_tokens"] = total_out

    result_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    job["result_df"] = result_df
    if not result_df.empty:
        xlsx_bytes = export_colored_xlsx(result_df)
        job["result_xlsx"] = xlsx_bytes
        try:
            RESULTS_DIR.mkdir(exist_ok=True)
            result_path = RESULTS_DIR / f"apollo_scored_{job_id[:8]}.xlsx"
            result_path.write_bytes(xlsx_bytes)
            job["result_path"] = str(result_path)
            job["log"].append(f"💾 Сохранено: {result_path.name}")
        except Exception:
            pass

    job["log"].append(
        "Готово." if not stop_event.is_set()
        else "Остановлено — частичный результат доступен."
    )
    job["done"] = True


def _start_job(api_key: str, icp_desc: str, df: pd.DataFrame) -> str:
    job_id = str(uuid.uuid4())
    _get_jobs()[job_id] = {
        "stop_event": threading.Event(),
        "done": False,
        "processed": 0,
        "total": len(df),
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "log": [],
        "result_df": None,
        "result_xlsx": None,
    }
    threading.Thread(target=_worker, args=(job_id, api_key, icp_desc, df), daemon=True).start()
    return job_id


def _get_job(job_id: str | None) -> dict[str, Any] | None:
    return _get_jobs().get(job_id) if job_id else None


# ── main entry point ──────────────────────────────────────────────────────────

def run() -> None:
    _init_session()

    st.title("Lead Scoring")
    st.caption("Apollo export → Claude Haiku 4.5 cascade: Apollo data → website → DuckDuckGo.")

    profiles = _load_profiles()
    profile_names = list(profiles.keys())
    api_key = _get_api_key()

    if not profile_names:
        st.error("Файл profiles.yaml не найден или пустой.")
        return

    # ── config row ────────────────────────────────────────────────────────────
    cfg_left, cfg_mid, cfg_right = st.columns([3, 2, 2])

    with cfg_left:
        uploaded = st.file_uploader(
            "Apollo export (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            disabled=bool(_get_job(st.session_state.ls_job_id) and not _get_job(st.session_state.ls_job_id)["done"]),
        )
        if uploaded is not None:
            try:
                st.session_state.ls_uploaded_df = (
                    pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv")
                    else pd.read_excel(uploaded)
                )
                st.session_state.ls_upload_name = uploaded.name
            except Exception as e:
                st.error(f"Ошибка чтения: {e}")
                st.session_state.ls_uploaded_df = None

        if st.session_state.ls_uploaded_df is not None:
            st.caption(f"✅ {st.session_state.ls_upload_name} — {len(st.session_state.ls_uploaded_df)} строк")

    with cfg_mid:
        default_idx = 0
        if st.session_state.ls_icp_profile in profile_names:
            default_idx = profile_names.index(st.session_state.ls_icp_profile)
        icp = st.selectbox("ICP профиль", profile_names, index=default_idx, key="ls_icp_select")
        st.session_state.ls_icp_profile = icp

    with cfg_right:
        st.number_input(
            "Input цена ($/1M токенов)",
            min_value=0.01, max_value=1000.0, step=0.1, format="%.2f",
            key="ls_cost_input",
            help="Haiku 4.5 = $1.00/M",
        )
        st.number_input(
            "Output цена ($/1M токенов)",
            min_value=0.01, max_value=1000.0, step=0.1, format="%.2f",
            key="ls_cost_output",
            help="Haiku 4.5 = $5.00/M",
        )

    st.divider()

    # ── START / STOP ──────────────────────────────────────────────────────────
    df = st.session_state.ls_uploaded_df
    icp_key = st.session_state.ls_icp_profile
    icp_desc = profiles[icp_key].get("description", "") if icp_key else ""
    job = _get_job(st.session_state.ls_job_id)
    is_running = bool(job and not job["done"])

    btn_col, msg_col = st.columns([2, 4])
    with btn_col:
        b_start = st.button(
            "▶ START SCORE",
            type="primary",
            disabled=is_running or df is None or not api_key,
            width="stretch",
        )
        b_stop = st.button("⏹ STOP", disabled=not is_running, width="stretch")

    with msg_col:
        if df is None:
            st.info("Загрузите CSV или Excel выше.")
        elif not api_key:
            st.error("ANTHROPIC_API_KEY не найден — добавьте в .streamlit/secrets.toml.")
        elif is_running:
            st.info("Идёт скоринг… нажмите ⏹ STOP чтобы прервать.")

    if b_start and df is not None and api_key:
        st.session_state.ls_job_id = _start_job(api_key, icp_desc, df)
        st.rerun()

    if b_stop and job:
        job["stop_event"].set()

    # ── dashboard ─────────────────────────────────────────────────────────────
    job = _get_job(st.session_state.ls_job_id)
    progress_bar = st.progress(0.0)
    c1, c2, c3, c4, c5 = st.columns(5)
    m_proc    = c1.empty()
    m_err     = c2.empty()
    m_in_tok  = c3.empty()
    m_out_tok = c4.empty()
    m_cost    = c5.empty()
    log_area  = st.empty()

    cost_in  = float(st.session_state.get("ls_cost_input") or DEFAULT_COST_INPUT_PER_MILLION_USD)
    cost_out = float(st.session_state.get("ls_cost_output") or DEFAULT_COST_OUTPUT_PER_MILLION_USD)

    def _render(j: dict) -> None:
        total = j["total"] or 1
        progress_bar.progress(min(j["processed"] / total, 1.0))
        m_proc.metric("Обработано", f"{j['processed']} / {j['total']}")
        m_err.metric("Ошибок", j["errors"])
        in_tok, out_tok = j["input_tokens"], j["output_tokens"]
        m_in_tok.metric("Input токены", f"{in_tok:,}".replace(",", "\u202f"))
        m_out_tok.metric("Output токены", f"{out_tok:,}".replace(",", "\u202f"))
        m_cost.metric("Затраты ($)", f"${estimate_cost_usd(in_tok, out_tok, cost_in, cost_out):.4f}")
        with log_area.container(height=320, border=True):
            st.caption("Лог обработки")
            st.code("\n".join(j["log"][-100:]), language=None)

    if job is None:
        for m in (m_proc, m_err, m_in_tok, m_out_tok, m_cost):
            m.metric("—", "—")
    elif not job["done"]:
        _render(job)
        time.sleep(0.6)
        st.rerun()
    else:
        _render(job)
        result_df   = job.get("result_df")
        result_xlsx = job.get("result_xlsx")
        if result_df is not None and not result_df.empty:
            st.success(f"Готово! Обработано {job['processed']} компаний.")
            if result_xlsx:
                st.download_button(
                    label="⬇ Скачать результат (.xlsx)",
                    data=result_xlsx,
                    file_name="apollo_scored.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{st.session_state.ls_job_id}",
                )

    # ── saved results ─────────────────────────────────────────────────────────
    if RESULTS_DIR.exists():
        saved = sorted(RESULTS_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if saved:
            st.divider()
            st.subheader("Сохранённые результаты")
            for i, path in enumerate(saved[:10]):
                ts = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
                col_name, col_dl, col_del = st.columns([4, 2, 1])
                col_name.write(f"📄 {path.name}  \n*{ts}*")
                with open(path, "rb") as f:
                    col_dl.download_button(
                        label="⬇ Скачать",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"saved_dl_{i}",
                    )
                if col_del.button("🗑", key=f"del_{i}", help="Удалить"):
                    path.unlink(missing_ok=True)
                    st.rerun()
