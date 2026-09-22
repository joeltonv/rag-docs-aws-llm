from __future__ import annotations

import streamlit as st

from streamlit_app.config import StreamlitAppConfig
from streamlit_app.components.rendering import render_benchmark_comparison
from streamlit_app.services.api_client import RagApiClient, RagApiClientError
from streamlit_app.services.benchmark import BenchmarkOutcome, run_benchmark


def render(client: RagApiClient, config: StreamlitAppConfig) -> None:
    st.title("Benchmark")

    with st.form("benchmark_form", clear_on_submit=False):
        question = st.text_area("Pergunta", height=120)
        available_models = {option.label: option for option in config.model_options}
        selected_labels = st.multiselect(
            "Modelos",
            options=list(available_models.keys()),
            default=list(available_models.keys()),
        )
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=config.default_top_k, step=1)
        submitted = st.form_submit_button("Executar benchmark", use_container_width=True, type="primary")

    if not submitted:
        return

    if not question.strip():
        st.warning("Informe uma pergunta antes de executar o benchmark.")
        return

    selected_models = tuple(available_models[label] for label in selected_labels)
    if not selected_models:
        st.warning("Selecione ao menos um modelo para a comparação.")
        return

    try:
        with st.spinner("Recuperando contexto e executando as comparações em paralelo..."):
            outcome = run_benchmark(
                client=client,
                question=question.strip(),
                top_k=int(top_k),
                models=selected_models,
            )
    except RagApiClientError as exc:
        st.error(str(exc))
        return

    _render_results(outcome)


def _render_results(outcome: BenchmarkOutcome) -> None:
    st.markdown(f"**Pergunta**\n\n{outcome.question}")
    render_benchmark_comparison(outcome.results)