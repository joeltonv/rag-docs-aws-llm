from __future__ import annotations

import streamlit as st

from streamlit_app.config import ModelOption, StreamlitAppConfig
from streamlit_app.components.rendering import render_compact_answer, render_context_block, render_sources_accordions
from streamlit_app.services.api_client import RagApiClient, RagApiClientError


def _default_model_index(options: tuple[ModelOption, ...], default_model_id: str) -> int:
    for index, option in enumerate(options):
        if option.model_id == default_model_id:
            return index
    return 0


def render(client: RagApiClient, config: StreamlitAppConfig) -> None:
    st.title("Consulta")

    with st.form("single_query_form", clear_on_submit=False):
        question = st.text_area("Pergunta", height=120)

        left_column, right_column = st.columns(2)
        with left_column:
            model_labels = [option.label for option in config.model_options]
            selected_label = st.selectbox(
                "Modelo",
                options=model_labels,
                index=_default_model_index(config.model_options, config.default_model_id),
            )

        with right_column:
            top_k = st.slider("Top-K", min_value=1, max_value=20, value=config.default_top_k, step=1)

        submitted = st.form_submit_button("Perguntar", use_container_width=True, type="primary")

    if not submitted:
        return

    if not question.strip():
        st.warning("Informe uma pergunta antes de executar a consulta.")
        return

    selected_model = next(option for option in config.model_options if option.label == selected_label)

    try:
        with st.spinner("Consultando o backend RAG..."):
            payload = client.chat(
                question=question.strip(),
                top_k=int(top_k),
                model_id=selected_model.model_id,
            )
    except RagApiClientError as exc:
        st.error(str(exc))
        return

    question_text = question.strip()
    answer_text = str(payload.get("answer") or "").strip()
    render_compact_answer(question_text, answer_text, payload.get("request_seconds"))
    render_sources_accordions(payload.get("results") or [])
    render_context_block(str(payload.get("context") or ""), key="single-query")