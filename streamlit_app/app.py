from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from streamlit_app.config import load_config
from streamlit_app.services.api_client import RagApiClient
from streamlit_app.state import initialize_state
from streamlit_app.ui.theme import inject_theme_css
from streamlit_app.views.benchmark import render as render_benchmark_page
from streamlit_app.views.single_query import render as render_single_query_page
from streamlit_app.views.status import render as render_status_page


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGE_OPTIONS = [
    ("Consulta", "Consulta"),
    ("Benchmark", "Benchmark"),
    ("Status", "Status"),
]


@st.cache_resource(show_spinner=False)
def _get_client(api_base_url: str, request_timeout_seconds: float) -> RagApiClient:
    return RagApiClient(base_url=api_base_url, timeout_seconds=request_timeout_seconds)


def main() -> None:
    initialize_state()
    config = load_config()
    st.set_page_config(page_title=config.app_title, page_icon="RAG", layout="wide", initial_sidebar_state="expanded")

    client = _get_client(config.api_base_url, config.request_timeout_seconds)
    selection = _render_sidebar()
    inject_theme_css()

    if selection == "Consulta":
        render_single_query_page(client, config)
    elif selection == "Benchmark":
        render_benchmark_page(client, config)
    elif selection == "Status":
        render_status_page(client, config)


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Navegação")
        selection = st.radio(
            "Seção",
            options=[label for label, _icon in PAGE_OPTIONS],
            label_visibility="collapsed",
        )

    return selection


if __name__ == "__main__":
    main()
