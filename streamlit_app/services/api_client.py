from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import requests


class RagApiClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagApiClient:
    base_url: str
    timeout_seconds: float

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, params: Mapping[str, str] | None = None) -> dict[str, Any]:
        started_at = perf_counter()
        query_params = {key: value for key, value in (params or {}).items() if value != ""}
        url = self._build_url(path)

        try:
            response = requests.request(
                method=method,
                url=url,
                params=query_params or None,
                json=payload,
                timeout=self.timeout_seconds,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload_data = response.json() if response.content else {}
            if isinstance(payload_data, dict):
                payload_data["request_seconds"] = perf_counter() - started_at
            return payload_data
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            if response is not None:
                detail = response.text
                status_code = response.status_code
                raise RagApiClientError(f"HTTP {status_code} ao chamar {path}: {detail}") from exc

            raise RagApiClientError(f"Falha ao chamar {path}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def search(self, question: str, top_k: int) -> dict[str, Any]:
        return self._request("GET", "/search", params={"q": question, "top_k": str(top_k)})

    def chat(
        self,
        question: str,
        top_k: int,
        model_id: str | None = None,
        context: str | None = None,
        sources: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if model_id:
            payload["model_id"] = model_id
        if context is not None:
            payload["context"] = context
        if sources is not None:
            payload["results"] = list(sources)

        return self._request("POST", "/chat", payload=payload)
