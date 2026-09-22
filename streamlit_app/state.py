from __future__ import annotations

import streamlit as st


NAV_STATE_KEY = "rag_navigation"


def initialize_state() -> None:
    st.session_state.setdefault(NAV_STATE_KEY, "Consulta")


def get_navigation() -> str:
    initialize_state()
    return str(st.session_state.get(NAV_STATE_KEY, "Consulta"))


def set_navigation(page: str) -> None:
    st.session_state[NAV_STATE_KEY] = page
