from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from experiments.config import RetrievalVariant


logger = logging.getLogger(__name__)


def _canonical_path_key(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip().lower()
    if not text:
        return ""

    if len(text) >= 3 and text[1:3] == ":/":
        text = text[3:]

    docs_index = text.find("docs_markdown/")
    if docs_index != -1:
        return text[docs_index:]

    return text.split("/")[-1]


class RagServiceProtocol(Protocol):
    def search(self, question: str, top_k: int | None = None, translate_query: bool | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def chat(
        self,
        question: str,
        top_k: int | None = None,
        model_id: str | None = None,
        context_override: str | None = None,
        sources_override: list[dict[str, Any]] | None = None,
        translate_query: bool | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class EvaluationOutput:
    raw_results_csv: Path
    records: list[dict[str, Any]]


def _distance_to_similarity(distance: float | None) -> float | None:
    if distance is None:
        return None

    return 1.0 / (1.0 + max(float(distance), 0.0))


def _extract_tokens(generation_metadata: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    input_tokens: int | None = None
    output_tokens: int | None = None

    usage = generation_metadata.get("usage")
    if isinstance(usage, dict):
        raw_input = usage.get("inputTokens", usage.get("input_tokens"))
        raw_output = usage.get("outputTokens", usage.get("output_tokens"))
        if isinstance(raw_input, int):
            input_tokens = raw_input
        if isinstance(raw_output, int):
            output_tokens = raw_output

    if input_tokens is None and isinstance(generation_metadata.get("prompt_token_count"), int):
        input_tokens = int(generation_metadata["prompt_token_count"])
    if output_tokens is None and isinstance(generation_metadata.get("generation_token_count"), int):
        output_tokens = int(generation_metadata["generation_token_count"])

    total_tokens: int | None = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    elif output_tokens is not None:
        total_tokens = output_tokens
    elif input_tokens is not None:
        total_tokens = input_tokens

    return input_tokens, output_tokens, total_tokens


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class ExperimentEvaluator:
    def __init__(
        self,
        rag_service: RagServiceProtocol,
        model_ids: tuple[str, ...],
        top_k: int = 10,
        retrieval_variants: tuple[RetrievalVariant, ...] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.rag_service = rag_service
        self.model_ids = model_ids
        self.top_k = top_k
        self.retrieval_variants = retrieval_variants or (
            RetrievalVariant(
                name="production",
                top_k=top_k,
                translate_query=True,
                description="Configuração única baseada no Top-K informado.",
            ),
        )
        self.logger = logger_ or logger

    def run(self, dataset: list[dict[str, Any]], output_csv: Path) -> EvaluationOutput:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []

        for example in dataset:
            question = str(example["pergunta"])
            ground_truth_source = str(example["source_path"])
            ground_truth_key = _canonical_path_key(ground_truth_source)
            for variant in self.retrieval_variants:
                search_result = self.rag_service.search(question, top_k=variant.top_k, translate_query=variant.translate_query)
                context = str(search_result.get("context", ""))
                retrieved_results = search_result.get("results") or []
                source_paths = [str(item.get("source_path", "")) for item in retrieved_results if isinstance(item, dict)]

                enriched_chunks: list[dict[str, Any]] = []
                similarity_scores: list[float] = []
                relevant_rank = None
                relevant_count = 0

                for index, chunk in enumerate(retrieved_results, start=1):
                    if not isinstance(chunk, dict):
                        continue

                    metadata = _as_dict(chunk.get("metadata"))
                    source_path = str(chunk.get("source_path") or metadata.get("source_path") or "")
                    distance = chunk.get("distance")
                    similarity = _distance_to_similarity(float(distance)) if isinstance(distance, (int, float)) else None
                    source_key = _canonical_path_key(source_path)

                    if similarity is not None:
                        similarity_scores.append(similarity)

                    if ground_truth_key and source_key and source_key == ground_truth_key:
                        relevant_count += 1
                        if relevant_rank is None:
                            relevant_rank = index

                    enriched_chunks.append(
                        {
                            "rank": index,
                            "source_path": source_path,
                            "titulo": chunk.get("titulo") or metadata.get("titulo"),
                            "hierarquia": chunk.get("hierarquia") or metadata.get("hierarquia"),
                            "distance": distance,
                            "similarity": similarity,
                            "document": chunk.get("document", ""),
                        }
                    )

                similarity_mean = sum(similarity_scores) / len(similarity_scores) if similarity_scores else None

                for model_id in self.model_ids:
                    started_at = time.perf_counter()
                    result = self.rag_service.chat(
                        question,
                        top_k=variant.top_k,
                        model_id=model_id,
                        context_override=context,
                        sources_override=retrieved_results,
                        translate_query=variant.translate_query,
                    )
                    elapsed_seconds = time.perf_counter() - started_at

                    generation_metadata = _as_dict(result.get("generation_metadata"))
                    input_tokens, output_tokens, total_tokens = _extract_tokens(generation_metadata)

                    record = {
                        "id": example.get("id"),
                        "pergunta": question,
                        "refraseamento": search_result.get("retrieval_question", question),
                        "variacao_retrieval": variant.name,
                        "variacao_descricao": variant.description,
                        "query_translation": _serialize_json(search_result.get("query_translation", {})),
                        "resposta_ideal": example.get("resposta_ideal", ""),
                        "source_path_ground_truth": ground_truth_source,
                        "titulo_documento": example.get("titulo_documento", ""),
                        "secao": example.get("secao", ""),
                        "categoria": example.get("categoria", ""),
                        "dificuldade": example.get("dificuldade", ""),
                        "modelo": model_id,
                        "resposta": result.get("answer", ""),
                        "tempo_resposta_s": round(elapsed_seconds, 6),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens": total_tokens,
                        "top_k": variant.top_k,
                        "translate_query": variant.translate_query,
                        "chunks_recuperados": len(enriched_chunks),
                        "chunks_recuperados_json": _serialize_json(enriched_chunks),
                        "source_paths_chunks": _serialize_json(source_paths),
                        "similaridades_chunks": _serialize_json([chunk["similarity"] for chunk in enriched_chunks]),
                        "similaridade_media_chunks": similarity_mean,
                        "retrieved_rank": relevant_rank,
                        "retrieval_hit": bool(relevant_rank and relevant_rank <= variant.top_k),
                        "hit_rate_at_k": 1.0 if relevant_rank is not None and relevant_rank <= variant.top_k else 0.0,
                        "recall_at_k": float(relevant_count > 0),
                        "precision_at_k": relevant_count / max(variant.top_k, 1),
                        "mrr": 1.0 / relevant_rank if relevant_rank is not None else 0.0,
                        "context_truncated": bool(result.get("context_truncated", False)),
                        "contexto": result.get("context", context),
                        "generation_metadata": _serialize_json(generation_metadata),
                        "answer_length": len(str(result.get("answer", ""))),
                        "question_length": len(question),
                    }
                    records.append(record)

                    self.logger.info(
                        "Avaliado exemplo %s com variante %s e modelo %s em %.2fs.",
                        example.get("id"),
                        variant.name,
                        model_id,
                        elapsed_seconds,
                    )

        self._write_csv(records, output_csv)
        return EvaluationOutput(raw_results_csv=output_csv, records=records)

    def _write_csv(self, records: list[dict[str, Any]], output_csv: Path) -> None:
        if not records:
            output_csv.write_text("", encoding="utf-8")
            return

        fieldnames = list(records[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
