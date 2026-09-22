from __future__ import annotations

from html import escape
from typing import Any, Sequence

import streamlit as st

from streamlit_app.services.benchmark import BenchmarkResult
from streamlit_app.utils.formatting import format_duration, format_token_summary


def render_hero(title: str, subtitle: str, badges: Sequence[str] | None = None) -> None:
    _ = badges
    st.markdown(
        f"""
                <div class="app-page-title">
                    <h1>{escape(title)}</h1>
                    <p>{escape(subtitle)}</p>
                </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(items: Sequence[tuple[str, str]]) -> None:
    if not items:
        return

    columns = st.columns(min(len(items), 4), gap="small")
    for index, (label, value) in enumerate(items):
        column = columns[index % len(columns)]
        with column:
            st.metric(label, value)


def render_info_cards(items: Sequence[tuple[str, str]]) -> None:
    if not items:
        return

    cards_html = []
    for label, value in items:
        cards_html.append(
            "<div class=\"app-status-card\">"
            f"<div class=\"app-status-card__label\">{escape(label)}</div>"
            f"<div class=\"app-status-card__value\">{escape(value)}</div>"
            "</div>"
        )

    st.markdown(
        f"<div class=\"app-status-grid\">{''.join(cards_html)}</div>",
        unsafe_allow_html=True,
    )


def render_sources_table(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        st.info("Nenhum documento foi recuperado.")
        return

    rows = []
    for item in results:
        rows.append(
            {
                "rank": item.get("rank"),
                "software": item.get("software"),
                "hierarquia": item.get("hierarquia"),
                "titulo": item.get("titulo"),
                "source_path": item.get("source_path"),
                "distance": item.get("distance"),
                "score": item.get("score"),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_sources_accordions(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        st.info("Nenhum chunk recuperado.")
        return

    for item in results:
        rank = item.get("rank", "?")
        title = item.get("titulo") or item.get("document", "Chunk")
        source_path = item.get("source_path", "N/D")
        summary = f"{item.get('software', 'N/D')} • {item.get('hierarquia', 'N/D')}"
        with st.expander(f"Trecho {rank} • {title}", expanded=False):
            st.caption(summary)
            st.write(item.get("document", ""))
            st.caption(f"Fonte: {source_path} • Distância: {item.get('distance', 'N/D')}")


def render_context_block(context: str, key: str = "context") -> None:
    _ = key
    with st.expander("Contexto enviado ao LLM", expanded=False):
        st.code(context or "", language="markdown")


def render_single_answer(payload: dict[str, Any]) -> None:
    answer = str(payload.get("answer", "")).strip()
    if answer:
        st.markdown(answer)
    else:
        st.warning("A resposta não veio preenchida.")


def render_chat_panel(question: str, answer: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.markdown(answer)


def render_compact_answer(question: str, answer: str, latency_seconds: float | None) -> None:
    st.markdown(f"**Pergunta**\n\n{question}")
    st.markdown(f"**Resposta**\n\n{answer or 'Sem resposta.'}")
    st.caption(f"Latência da consulta: {format_duration(latency_seconds)}")


def render_benchmark_card(result: BenchmarkResult) -> None:
    st.markdown(f"#### {escape(result.label)}", unsafe_allow_html=True)
    st.caption(result.model_id)

    if result.error:
        st.error(result.error)
        return

    st.markdown(result.answer or "Sem resposta.")
    st.markdown(
        f"**Tempo:** {format_duration(result.latency_seconds)}  \n**Tokens:** {format_token_summary(result.tokens)}",
    )

    if result.metrics:
        with st.expander("Métricas", expanded=False):
            st.json(result.metrics)


def render_benchmark_comparison(results: Sequence[BenchmarkResult]) -> None:
    if not results:
        st.info("Nenhum resultado foi retornado.")
        return

    columns = st.columns(len(results), gap="medium")
    for column, result in zip(columns, results):
        with column:
            st.markdown(f"### {escape(result.label)}")
            st.caption(result.model_id)
            if result.error:
                st.error(result.error)
            else:
                st.markdown(result.answer or "Sem resposta.")
            st.caption(f"Tempo de resposta: {format_duration(result.latency_seconds)}")
