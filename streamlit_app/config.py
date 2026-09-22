from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelOption:
    label: str
    model_id: str


@dataclass(frozen=True)
class StreamlitAppConfig:
    app_title: str
    api_base_url: str
    request_timeout_seconds: float
    default_top_k: int
    default_model_id: str
    model_options: tuple[ModelOption, ...]
    models_file: Path


DEFAULT_MODEL_OPTIONS: tuple[ModelOption, ...] = (
    ModelOption("Meta Llama 4 Maverick 17B Instruct", "us.meta.llama4-maverick-17b-instruct-v1:0"),
    ModelOption("DeepSeek V3.2", "deepseek.v3.2"),
    ModelOption("Mistral Large 3 675B Instruct", "mistral.mistral-large-3-675b-instruct"),
)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if value is not None else default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _parse_model_mapping(item: Any) -> ModelOption | None:
    if not isinstance(item, dict):
        return None

    label = str(item.get("label", "")).strip()
    model_id = str(item.get("model_id", "")).strip()
    if not label or not model_id:
        return None

    return ModelOption(label=label, model_id=model_id)


def load_model_options(models_file: Path) -> tuple[ModelOption, ...]:
    inline_json = os.getenv("STREAMLIT_MODELS_JSON")
    if inline_json:
        try:
            parsed = json.loads(inline_json)
        except json.JSONDecodeError:
            parsed = []

        if isinstance(parsed, list):
            options = tuple(option for option in (_parse_model_mapping(item) for item in parsed) if option is not None)
            if options:
                return options

    if models_file.exists():
        try:
            parsed = json.loads(models_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = []

        if isinstance(parsed, list):
            options = tuple(option for option in (_parse_model_mapping(item) for item in parsed) if option is not None)
            if options:
                return options

    return DEFAULT_MODEL_OPTIONS


def load_config() -> StreamlitAppConfig:
    repo_root = Path(__file__).resolve().parents[1]
    models_file = Path(_env_str("STREAMLIT_MODELS_FILE", str(repo_root / "streamlit_app" / "models.json")))
    model_options = load_model_options(models_file)
    default_model_id = _env_str("STREAMLIT_DEFAULT_MODEL_ID", model_options[0].model_id)

    return StreamlitAppConfig(
        app_title=_env_str("STREAMLIT_APP_TITLE", "RAG Demo - TCC Pós LLMs"),
        api_base_url=_env_str("STREAMLIT_API_BASE_URL", "http://127.0.0.1:5000"),
        request_timeout_seconds=_env_float("STREAMLIT_REQUEST_TIMEOUT_SECONDS", 120.0),
        default_top_k=_env_int("STREAMLIT_DEFAULT_TOP_K", 4),
        default_model_id=default_model_id,
        model_options=model_options,
        models_file=models_file,
    )
