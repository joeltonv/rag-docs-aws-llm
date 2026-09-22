from __future__ import annotations

from typing import Any, Iterable


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/D"
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    return f"{minutes}m {remaining:.1f}s"


def format_token_summary(tokens: dict[str, Any] | None) -> str:
    if not tokens:
        return "N/D"

    input_tokens = tokens.get("inputTokens") or tokens.get("input_tokens")
    output_tokens = tokens.get("outputTokens") or tokens.get("output_tokens")
    total_tokens = tokens.get("totalTokens") or tokens.get("total_tokens")

    parts: list[str] = []
    if input_tokens is not None:
        parts.append(f"entrada: {input_tokens}")
    if output_tokens is not None:
        parts.append(f"saída: {output_tokens}")
    if total_tokens is not None:
        parts.append(f"total: {total_tokens}")

    return " | ".join(parts) if parts else "N/D"


def ensure_mapping_list(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Iterable):
        return []

    return [item for item in values if isinstance(item, dict)]
