from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

from streamlit_app.config import ModelOption
from streamlit_app.services.api_client import RagApiClient


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    model_id: str
    answer: str | None
    latency_seconds: float
    chunk_count: int
    tokens: dict[str, Any] | None
    metrics: dict[str, Any] | None
    error: str | None
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkOutcome:
    question: str
    top_k: int
    search_payload: dict[str, Any]
    results: tuple[BenchmarkResult, ...]


def _extract_tokens(payload: dict[str, Any]) -> dict[str, Any] | None:
    generation = payload.get("generation_metadata")
    if not isinstance(generation, dict):
        return None

    usage = generation.get("usage")
    if isinstance(usage, dict):
        return usage

    return None


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any] | None:
    generation = payload.get("generation_metadata")
    if not isinstance(generation, dict):
        return None

    metrics = generation.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    return None


def _run_single_model(
    client: RagApiClient,
    question: str,
    top_k: int,
    model: ModelOption,
    context: str,
    sources: Sequence[dict[str, Any]],
) -> BenchmarkResult:
    started_at = perf_counter()
    try:
        payload = client.chat(
            question=question,
            top_k=top_k,
            model_id=model.model_id,
            context=context,
            sources=sources,
        )
        latency_seconds = perf_counter() - started_at
        return BenchmarkResult(
            label=model.label,
            model_id=model.model_id,
            answer=str(payload.get("answer", "")).strip() or None,
            latency_seconds=latency_seconds,
            chunk_count=len(payload.get("results") or sources),
            tokens=_extract_tokens(payload),
            metrics=_extract_metrics(payload),
            error=None,
            raw_response=payload,
        )
    except Exception as exc:
        latency_seconds = perf_counter() - started_at
        return BenchmarkResult(
            label=model.label,
            model_id=model.model_id,
            answer=None,
            latency_seconds=latency_seconds,
            chunk_count=len(sources),
            tokens=None,
            metrics=None,
            error=str(exc),
            raw_response={},
        )


def run_benchmark(
    client: RagApiClient,
    question: str,
    top_k: int,
    models: Sequence[ModelOption],
) -> BenchmarkOutcome:
    search_payload = client.search(question=question, top_k=top_k)
    context = str(search_payload.get("context", ""))
    raw_sources = search_payload.get("results") or []
    sources = tuple(item for item in raw_sources if isinstance(item, dict))
    effective_top_k = int(search_payload.get("top_k") or top_k)

    if not models:
        return BenchmarkOutcome(question=question, top_k=effective_top_k, search_payload=search_payload, results=())

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(_run_single_model, client, question, effective_top_k, model, context, sources): model
            for model in models
        }
        collected: list[BenchmarkResult] = []
        for future in as_completed(futures):
            collected.append(future.result())

    collected.sort(key=lambda item: next(index for index, model in enumerate(models) if model.model_id == item.model_id))
    return BenchmarkOutcome(question=question, top_k=effective_top_k, search_payload=search_payload, results=tuple(collected))
