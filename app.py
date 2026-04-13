import streamlit as st

st.set_page_config(page_title="LeadGen Suite", layout="wide", page_icon="🎯")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }

        div[data-testid="stHorizontalBlock"] button {
            border-radius: 999px;
            padding: 0.35rem 1.25rem;
            font-weight: 500;
            transition: background 0.15s;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

TOOLS = [
    "Lead Scoring",
    "Apollo Cleaner",
]

if "active_tool" not in st.session_state:
    st.session_state.active_tool = TOOLS[0]

cols = st.columns(len(TOOLS), gap="small")
for col, label in zip(cols, TOOLS):
    with col:
        is_active = st.session_state.active_tool == label
        if st.button(
            label,
            key=f"nav_{label}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_tool = label
            st.rerun()

st.divider()

active = st.session_state.active_tool

if active == "Lead Scoring":
    from pages.lead_scoring import run
    run()
elif active == "Apollo Cleaner":
    from pages.apollo_cleaner import run
    run()
