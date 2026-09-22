from __future__ import annotations

from pathlib import Path

import streamlit as st

from streamlit_app.components.rendering import render_info_cards
from streamlit_app.config import StreamlitAppConfig
from streamlit_app.services.api_client import RagApiClient, RagApiClientError


def render(client: RagApiClient, config: StreamlitAppConfig) -> None:
    st.title("Status")

    try:
        payload = client.health()
    except RagApiClientError as exc:
        st.error(str(exc))
        return

    if payload.get("status") != "ok":
        st.error(str(payload.get("error") or "A API respondeu em modo degraded."))
    else:
        st.success("API disponível.")

    api_model_id = str(payload.get("selected_model_id") or payload.get("chat_model_id") or "N/D")
    ui_default_model_id = config.default_model_id
    embedding_model_raw = str(payload.get("embedding_model") or "N/D")
    embedding_model_name = Path(embedding_model_raw).name if embedding_model_raw != "N/D" else "N/D"
    embedding_provider = str(payload.get("embedding_model_provider") or payload.get("embedding_provider") or "N/D")
    embedding_dimension = str(payload.get("embedding_model_dimension") or "N/D")
    region_name = str(payload.get("region_name") or "N/D")
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    environment_label = "Local" if embedding_provider == "sentence-transformers" else "Nuvem"

    infra_col, data_col, llm_col = st.columns(3, gap="medium")

    with infra_col:
        st.subheader("Infraestrutura")
        st.metric("Status API", "Online" if payload.get("status") == "ok" else "Erro")
        st.metric("Região AWS", region_name)
        st.metric("Ambiente", environment_label)

    with data_col:
        st.subheader("Base de Conhecimento")
        st.metric("Total de Chunks", str(payload.get("chunks_indexed", "N/D")))
        st.metric("Modelo Embeddings", embedding_model_name)
        st.caption(f"Provedor: {embedding_provider} | Dimensão: {embedding_dimension}")

    with llm_col:
        st.subheader("LLM Ativo")
        st.metric("Modelo Chat", api_model_id)
        if temperature is not None:
            st.write(f"**Temperature:** {temperature}")
        else:
            st.write("**Temperature:** N/D")
        if top_p is not None:
            st.write(f"**Top P:** {top_p}")
        else:
            st.write("**Top P:** N/D")
        st.caption(f"Modelo padrão da interface: {ui_default_model_id}")

    st.divider()
    render_info_cards([
        ("Coleção Chroma", str(payload.get("collection_name", "N/D"))),
    ])
