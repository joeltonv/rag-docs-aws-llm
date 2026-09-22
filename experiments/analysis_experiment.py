from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


NUMERIC_METRICS = [
    "tempo_resposta_s",
    "input_tokens",
    "output_tokens",
    "tokens",
    "answer_length",
    "question_length",
    "chunks_recuperados",
    "similaridade_media_chunks",
    "retrieved_rank",
    "hit_rate_at_1",
    "hit_rate_at_3",
    "hit_rate_at_4",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "groundedness_score",
    "corretude_score",
    "completude_score",
    "clareza_score",
    "precisao_tecnica_score",
    "alucinacao_score",
    "overall_retrieval_score",
    "overall_generation_score",
    "overall_score",
]


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


class JudgeServiceProtocol(Protocol):
    def generate_tool_use_with_metadata(
        self,
        prompt: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class SummaryRow:
    modelo: str
    metric: str
    mean: float
    ci95_low: float
    ci95_high: float
    std: float
    n: int


def _ensure_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        float_value = float(value)
        if not math.isfinite(float_value):
            return None
        return float_value
    except (TypeError, ValueError):
        return None


def _ensure_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _mean_confidence_interval(values: list[float]) -> tuple[float, float, float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    mean_value = statistics.mean(values)
    if len(values) < 2:
        return mean_value, mean_value, mean_value, 0.0

    std_value = statistics.stdev(values)
    margin = 1.96 * std_value / math.sqrt(len(values))
    return mean_value, mean_value - margin, mean_value + margin, std_value


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_df = pd.DataFrame(raw_dataset)
    if "id" not in dataset_df.columns:
        raise ValueError("Dataset precisa conter a coluna 'id'.")
    return dataset_df.copy()


def _load_rows(results_csv: Path) -> pd.DataFrame:
    return pd.read_csv(results_csv)


def _parse_json_field(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _judge_prompt(question: str, ideal_answer: str, generated_answer: str) -> str:
    return (
        "Avalie a resposta gerada em comparação com a resposta ideal para uma tarefa de RAG.\n\n"
        f"Pergunta: {question}\n\n"
        f"Resposta ideal:\n{ideal_answer}\n\n"
        f"Resposta gerada:\n{generated_answer}\n\n"
        "Retorne SOMENTE um JSON válido com a estrutura abaixo:\n"
        "{\n"
        '  "groundedness": {"score": 1, "comment": ""},\n'
        '  "corretude": {"score": 1, "comment": ""},\n'
        '  "completude": {"score": 1, "comment": ""},\n'
        '  "clareza": {"score": 1, "comment": ""},\n'
        '  "precisao_tecnica": {"score": 1, "comment": ""},\n'
        '  "alucinacao": {"score": 1, "comment": ""}\n'
        "}\n\n"
        "Use a escala de 1 a 5, onde 5 e melhor. Para alucinacao, 5 significa ausencia de alucinacao e 1 indica forte alucinacao."
    )


def _judge_tool_schema() -> dict[str, Any]:
    metrics = ("groundedness", "corretude", "completude", "clareza", "precisao_tecnica", "alucinacao")
    return {
        "type": "object",
        "properties": {
            metric: {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "comment": {"type": "string"},
                },
                "required": ["score", "comment"],
                "additionalProperties": False,
            }
            for metric in metrics
        },
        "required": list(metrics),
        "additionalProperties": False,
    }


def _score_payload(scores: dict[str, Any], key: str) -> tuple[int | None, str]:
    entry = scores.get(key, {})
    if not isinstance(entry, dict):
        return None, ""

    raw_score = entry.get("score")
    raw_comment = entry.get("comment", "")
    score = int(raw_score) if isinstance(raw_score, int) else None
    return score, str(raw_comment)


def _judge_record(judge_service: JudgeServiceProtocol, judge_model_id: str, question: str, ideal_answer: str, generated_answer: str, system_prompt: str) -> dict[str, Any]:
    prompt = _judge_prompt(question, ideal_answer, generated_answer)
    response = judge_service.generate_tool_use_with_metadata(
        prompt,
        tool_name="avaliar_resposta_rag",
        tool_description="Preenche uma avaliação estruturada de respostas de RAG com notas de 1 a 5.",
        input_schema=_judge_tool_schema(),
        system_prompt=system_prompt,
        model_id=judge_model_id,
    )
    parsed = response.get("payload", {})
    if not isinstance(parsed, dict):
        raise ValueError("Resposta do juiz nao contem um payload estruturado valido.")

    answer_text = str(response.get("answer", json.dumps(parsed, ensure_ascii=False)))

    record: dict[str, Any] = {
        "judge_model_id": response.get("model_id", judge_model_id),
        "judge_raw_answer": answer_text,
    }
    for metric in ("groundedness", "corretude", "completude", "clareza", "precisao_tecnica", "alucinacao"):
        score, comment = _score_payload(parsed, metric)
        record[f"{metric}_score"] = score
        record[f"{metric}_comment"] = comment

    return record


def _build_retrieval_metrics(row: pd.Series, dataset_row: pd.Series) -> dict[str, Any]:
    retrieved_results = _parse_json_field(row.get("chunks_recuperados_json"), [])
    similarity_values = [value for value in _parse_json_field(row.get("similaridades_chunks"), []) if isinstance(value, (int, float))]
    source_paths = _parse_json_field(row.get("source_paths_chunks"), [])
    if not isinstance(source_paths, list):
        source_paths = []

    ground_truth_source = str(dataset_row.get("source_path", ""))
    ground_truth_key = _canonical_path_key(ground_truth_source)
    relevant_rank = None
    relevant_count = 0

    for index, item in enumerate(retrieved_results, start=1):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path", ""))
        if _canonical_path_key(source_path) == ground_truth_key and ground_truth_key:
            relevant_count += 1
            if relevant_rank is None:
                relevant_rank = index

    top_k = _ensure_int(row.get("top_k")) or len(retrieved_results) or 1
    top_k = max(top_k, 1)

    hit_at_1 = 1.0 if relevant_rank == 1 else 0.0
    hit_at_3 = 1.0 if relevant_rank is not None and relevant_rank <= 3 else 0.0
    hit_at_4 = 1.0 if relevant_rank is not None and relevant_rank <= 4 else 0.0
    recall_at_k = float(relevant_count > 0)
    precision_at_k = relevant_count / max(top_k, 1)
    mrr = 1.0 / relevant_rank if relevant_rank is not None else 0.0
    similarity_mean = statistics.mean(similarity_values) if similarity_values else _ensure_float(row.get("similaridade_media_chunks"))

    return {
        "retrieval_hit": bool(relevant_rank),
        "retrieved_rank": relevant_rank,
        "hit_rate_at_1": hit_at_1,
        "hit_rate_at_3": hit_at_3,
        "hit_rate_at_4": hit_at_4,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "similaridade_media_chunks": similarity_mean,
        "source_paths_chunks": json.dumps(source_paths, ensure_ascii=False),
    }


def _normalized_generation_score(row: pd.Series) -> float | None:
    scores = []
    for metric in ("groundedness_score", "corretude_score", "completude_score", "clareza_score", "precisao_tecnica_score", "alucinacao_score"):
        value = _ensure_float(row.get(metric))
        if value is not None:
            scores.append(value / 5.0)
    return statistics.mean(scores) if scores else None


def _normalized_retrieval_score(row: pd.Series) -> float | None:
    scores = []
    for metric in ("hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_4", "recall_at_k", "precision_at_k", "mrr", "similaridade_media_chunks"):
        value = _ensure_float(row.get(metric))
        if value is not None:
            scores.append(max(0.0, min(1.0, value)))
    return statistics.mean(scores) if scores else None


def _final_overall_score(row: pd.Series) -> float | None:
    scores = [value for value in (_ensure_float(row.get("overall_retrieval_score")), _ensure_float(row.get("overall_generation_score"))) if value is not None]
    if not scores:
        return None
    return statistics.mean(scores)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: Figure, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def _make_bar_chart(title: str, labels: list[str], values: list[float], ylabel: str, figures_dir: Path, stem: str, color: str = "#0b5fff") -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, figures_dir, stem)


def _make_grouped_bar_chart(title: str, labels: list[str], series: dict[str, list[float]], ylabel: str, figures_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    offsets = np.linspace(-(len(series) - 1) / 2 * width, (len(series) - 1) / 2 * width, len(series))

    for offset, (name, values) in zip(offsets, series.items()):
        ax.bar(x + offset, values, width=width, label=name)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, figures_dir, stem)


def _make_boxplot(title: str, data: list[list[float]], labels: list[str], ylabel: str, figures_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, figures_dir, stem)


def _make_histogram(title: str, values: list[float], xlabel: str, figures_dir: Path, stem: str, bins: int = 20) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(values, bins=bins, color="#0b5fff", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequencia")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, figures_dir, stem)


def _make_scatter(title: str, x: list[float], y: list[float], xlabel: str, ylabel: str, figures_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, alpha=0.72, color="#0b5fff", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return _save_figure(fig, figures_dir, stem)


def _make_heatmap(title: str, matrix: pd.DataFrame, figures_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix.iat[row_index, col_index]
            if pd.notna(value):
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#111827")
    fig.colorbar(heatmap, ax=ax, label="Correlação")
    return _save_figure(fig, figures_dir, stem)


def _make_radar_chart(title: str, labels: list[str], series: dict[str, list[float]], figures_dir: Path, stem: str) -> list[Path]:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for name, values in series.items():
        plotted = values + values[:1]
        ax.plot(angles, plotted, linewidth=2, label=name)
        ax.fill(angles, plotted, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=20)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    return _save_figure(fig, figures_dir, stem)


def _compute_summary(rows: list[dict[str, Any]], model_ids: list[str], metrics: list[str]) -> list[SummaryRow]:
    summary_rows: list[SummaryRow] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in rows:
        model = str(row.get("modelo", ""))
        for metric in metrics:
            value = _ensure_float(row.get(metric))
            if value is not None:
                grouped[(model, metric)].append(value)

    for model in model_ids:
        for metric in metrics:
            values = grouped.get((model, metric), [])
            mean_value, low_value, high_value, std_value = _mean_confidence_interval(values)
            summary_rows.append(
                SummaryRow(
                    modelo=model,
                    metric=metric,
                    mean=mean_value,
                    ci95_low=low_value,
                    ci95_high=high_value,
                    std=std_value,
                    n=len(values),
                )
            )

    return summary_rows


def _wide_model_matrix(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    metrics = [metric for metric in NUMERIC_METRICS if metric in df.columns]
    grouped = df.groupby("modelo")[metrics].mean(numeric_only=True).reindex(model_order)
    grouped.index.name = "modelo"
    return grouped.reset_index()


def _case_rows(df: pd.DataFrame, case_type: str, subset: pd.DataFrame, sort_columns: list[str], ascending: list[bool], limit: int = 5) -> list[dict[str, Any]]:
    if subset.empty:
        return []

    ordered = subset.sort_values(sort_columns, ascending=ascending).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        rows.append(
            {
                "case_type": case_type,
                "id": row.get("id"),
                "modelo": row.get("modelo"),
                "pergunta": row.get("pergunta"),
                "resposta": row.get("resposta"),
                "resposta_ideal": row.get("resposta_ideal"),
                "retrieved_rank": row.get("retrieved_rank"),
                "retrieval_hit": row.get("retrieval_hit"),
                "similaridade_media_chunks": row.get("similaridade_media_chunks"),
                "overall_score": row.get("overall_score"),
                "overall_retrieval_score": row.get("overall_retrieval_score"),
                "overall_generation_score": row.get("overall_generation_score"),
                "groundedness_score": row.get("groundedness_score"),
                "corretude_score": row.get("corretude_score"),
                "completude_score": row.get("completude_score"),
                "clareza_score": row.get("clareza_score"),
                "precisao_tecnica_score": row.get("precisao_tecnica_score"),
                "alucinacao_score": row.get("alucinacao_score"),
                "tempo_resposta_s": row.get("tempo_resposta_s"),
                "tokens": row.get("tokens"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "chunks_recuperados": row.get("chunks_recuperados"),
                "context_truncated": row.get("context_truncated"),
                "no_context_recovered": row.get("no_context_recovered"),
                "potential_hallucination": row.get("potential_hallucination"),
                "note": case_type,
            }
        )
    return rows


def _write_markdown_report(output_path: Path, title: str, sections: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join([f"# {title}", "", *sections]).rstrip() + "\n", encoding="utf-8")


class ExperimentMetricsPipeline:
    def __init__(self, judge_service: JudgeServiceProtocol | None, judge_model_id: str, judge_system_prompt: str, logger_: logging.Logger | None = None) -> None:
        self.judge_service = judge_service
        self.judge_model_id = judge_model_id
        self.judge_system_prompt = judge_system_prompt
        self.logger = logger_ or logger

    def run(self, dataset_path: Path, results_csv: Path, consolidated_csv: Path, report_markdown: Path, figures_dir: Path) -> dict[str, Any]:
        dataset_df = _load_dataset(dataset_path)
        results_df = _load_rows(results_csv)

        dataset_lookup = dataset_df.set_index("id")
        consolidated_rows: list[dict[str, Any]] = []
        judge_disabled_due_to_error = False

        for _, row in results_df.iterrows():
            example_lookup = dataset_lookup.loc[str(row.get("id"))] if str(row.get("id")) in dataset_lookup.index else None
            if example_lookup is None:
                self.logger.warning("Exemplo %s nao encontrado no dataset; linha ignorada.", row.get("id"))
                continue

            if isinstance(example_lookup, pd.DataFrame):
                example = example_lookup.iloc[0]
            else:
                example = example_lookup

            retrieval_metrics = _build_retrieval_metrics(row, example)
            consolidated: dict[str, Any] = {str(key): value for key, value in row.to_dict().items()}
            consolidated.update(retrieval_metrics)

            if self.judge_service is not None and not judge_disabled_due_to_error:
                try:
                    judge_metrics = _judge_record(
                        self.judge_service,
                        self.judge_model_id,
                        str(row.get("pergunta", "")),
                        str(row.get("resposta_ideal", example.get("resposta_ideal", ""))),
                        str(row.get("resposta", "")),
                        self.judge_system_prompt,
                    )
                    consolidated.update(judge_metrics)
                except Exception as exc:
                    judge_disabled_due_to_error = True
                    self.logger.warning("Desativando o juiz automatico por falha na avaliacao: %s", exc)
                    for metric in ("groundedness", "corretude", "completude", "clareza", "precisao_tecnica", "alucinacao"):
                        consolidated[f"{metric}_score"] = None
                        consolidated[f"{metric}_comment"] = ""
                    consolidated["judge_model_id"] = ""
                    consolidated["judge_raw_answer"] = ""
            else:
                for metric in ("groundedness", "corretude", "completude", "clareza", "precisao_tecnica", "alucinacao"):
                    consolidated[f"{metric}_score"] = None
                    consolidated[f"{metric}_comment"] = ""
                consolidated["judge_model_id"] = ""
                consolidated["judge_raw_answer"] = ""

            consolidated["overall_retrieval_score"] = _normalized_retrieval_score(pd.Series(consolidated))
            consolidated["overall_generation_score"] = _normalized_generation_score(pd.Series(consolidated))
            consolidated["overall_score"] = _final_overall_score(pd.Series(consolidated))
            consolidated_rows.append(consolidated)

        consolidated_df = pd.DataFrame(consolidated_rows)
        if consolidated_df.empty:
            raise ValueError("Nao ha linhas validas para consolidar.")

        for column in NUMERIC_METRICS:
            if column in consolidated_df.columns:
                consolidated_df[column] = pd.to_numeric(consolidated_df[column], errors="coerce")
                consolidated_df[column] = consolidated_df[column].replace([np.inf, -np.inf], np.nan)

        consolidated_df["response_length_chars"] = consolidated_df["resposta"].fillna("").astype(str).str.len()
        consolidated_df["response_length_words"] = consolidated_df["resposta"].fillna("").astype(str).str.count(r"\S+")
        consolidated_df["question_length_words"] = consolidated_df["pergunta"].fillna("").astype(str).str.count(r"\S+")
        consolidated_df["has_context"] = consolidated_df["contexto"].fillna("").astype(str).str.strip().ne("")
        consolidated_df["no_context_recovered"] = consolidated_df["chunks_recuperados"].fillna(0).astype(float).eq(0) | ~consolidated_df["has_context"]
        consolidated_df["retrieval_failure"] = ~consolidated_df["retrieval_hit"].fillna(False).astype(bool)
        consolidated_df["potential_hallucination"] = False
        if consolidated_df["groundedness_score"].notna().any() or consolidated_df["alucinacao_score"].notna().any():
            consolidated_df["potential_hallucination"] = (
                consolidated_df["groundedness_score"].fillna(5).astype(float).le(2)
                | consolidated_df["alucinacao_score"].fillna(5).astype(float).le(2)
                | consolidated_df["corretude_score"].fillna(5).astype(float).le(2)
            )
        consolidated_df["correct_with_weak_retrieval"] = (
            consolidated_df["retrieval_hit"].fillna(False).astype(bool).eq(False)
            & consolidated_df["overall_generation_score"].fillna(0).astype(float).ge(0.8)
        )
        consolidated_df["response_words_per_token"] = consolidated_df["response_length_words"].where(consolidated_df["tokens"].fillna(0).gt(0)) / consolidated_df["tokens"].replace(0, np.nan)

        consolidated_df.to_csv(consolidated_csv, index=False, encoding="utf-8")

        analysis_dir = report_markdown.parent
        figures_dir.mkdir(parents=True, exist_ok=True)

        model_order = list(consolidated_df.groupby("modelo")["overall_score"].mean().sort_values(ascending=False).index)
        if not model_order:
            model_order = sorted(consolidated_df["modelo"].dropna().unique().tolist())

        metrics_to_summarize = [
            "hit_rate_at_1",
            "hit_rate_at_3",
            "hit_rate_at_4",
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "similaridade_media_chunks",
            "groundedness_score",
            "corretude_score",
            "completude_score",
            "clareza_score",
            "precisao_tecnica_score",
            "alucinacao_score",
            "overall_retrieval_score",
            "overall_generation_score",
            "overall_score",
        ]
        summary_records: list[dict[str, Any]] = [
            {str(key): value for key, value in record.items()}
            for record in consolidated_df.to_dict(orient="records")
        ]
        summary_rows = _compute_summary(summary_records, model_order, metrics_to_summarize)

        summary_csv_path = analysis_dir / "resumo_metricas.csv"
        _write_csv(
            summary_csv_path,
            [
                {
                    "modelo": row.modelo,
                    "metric": row.metric,
                    "mean": row.mean,
                    "ci95_low": row.ci95_low,
                    "ci95_high": row.ci95_high,
                    "std": row.std,
                    "n": row.n,
                }
                for row in summary_rows
            ],
        )

        wide_matrix = _wide_model_matrix(consolidated_df, model_order)
        matrix_csv_path = analysis_dir / "matriz_consolidada_por_modelo.csv"
        wide_matrix.to_csv(matrix_csv_path, index=False, encoding="utf-8")

        ranking_df = wide_matrix.copy()
        ranking_df["rank"] = ranking_df["overall_score"].rank(method="dense", ascending=False).astype(int)
        ranking_df = ranking_df.sort_values(["rank", "modelo"])
        ranking_csv_path = analysis_dir / "ranking_modelos.csv"
        ranking_df.to_csv(ranking_csv_path, index=False, encoding="utf-8")

        general_stats = {
            "numero_perguntas": int(consolidated_df["id"].nunique()),
            "numero_modelos_avaliados": int(consolidated_df["modelo"].nunique()),
            "numero_respostas": int(len(consolidated_df)),
            "tempo_medio_global_s": float(consolidated_df["tempo_resposta_s"].mean()),
            "tokens_entrada_total": float(consolidated_df["input_tokens"].fillna(0).sum()),
            "tokens_saida_total": float(consolidated_df["output_tokens"].fillna(0).sum()),
            "tokens_totais_total": float(consolidated_df["tokens"].fillna(0).sum()),
            "resposta_media_chars": float(consolidated_df["response_length_chars"].mean()),
            "resposta_media_words": float(consolidated_df["response_length_words"].mean()),
            "tempo_medio_por_modelo_s": float(consolidated_df.groupby("modelo")["tempo_resposta_s"].mean().mean()),
        }

        summary_by_model = consolidated_df.groupby("modelo").agg(
            n_respostas=("id", "count"),
            n_questoes=("id", "nunique"),
            tempo_medio_resposta_s=("tempo_resposta_s", "mean"),
            tempo_mediano_resposta_s=("tempo_resposta_s", "median"),
            input_tokens_medio=("input_tokens", "mean"),
            output_tokens_medio=("output_tokens", "mean"),
            tokens_medios=("tokens", "mean"),
            tokens_totais=("tokens", "sum"),
            resposta_media_chars=("response_length_chars", "mean"),
            resposta_media_words=("response_length_words", "mean"),
            similaridade_media_chunks=("similaridade_media_chunks", "mean"),
            retrieval_hit_rate=("retrieval_hit", "mean"),
            overall_retrieval_score=("overall_retrieval_score", "mean"),
            overall_generation_score=("overall_generation_score", "mean"),
            overall_score=("overall_score", "mean"),
            groundedness_score=("groundedness_score", "mean"),
            corretude_score=("corretude_score", "mean"),
            completude_score=("completude_score", "mean"),
            clareza_score=("clareza_score", "mean"),
            precisao_tecnica_score=("precisao_tecnica_score", "mean"),
            alucinacao_score=("alucinacao_score", "mean"),
            no_context_recovered=("no_context_recovered", "mean"),
            retrieval_failure=("retrieval_failure", "mean"),
            potential_hallucination=("potential_hallucination", "mean"),
            correct_with_weak_retrieval=("correct_with_weak_retrieval", "mean"),
        ).reset_index()

        resultados_resumidos_csv = analysis_dir / "resultados_resumidos.csv"
        summary_by_model.to_csv(resultados_resumidos_csv, index=False, encoding="utf-8")

        interesting_cases = []
        interesting_cases.extend(_case_rows(consolidated_df, "melhores_respostas", consolidated_df.sort_values("overall_score", ascending=False), ["overall_score"], [False]))
        interesting_cases.extend(_case_rows(consolidated_df, "piores_respostas", consolidated_df.sort_values("overall_score", ascending=True), ["overall_score"], [True]))
        interesting_cases.extend(_case_rows(consolidated_df, "maiores_similaridades", consolidated_df.sort_values("similaridade_media_chunks", ascending=False), ["similaridade_media_chunks"], [False]))
        interesting_cases.extend(_case_rows(consolidated_df, "menores_similaridades", consolidated_df.sort_values("similaridade_media_chunks", ascending=True), ["similaridade_media_chunks"], [True]))
        interesting_cases.extend(_case_rows(consolidated_df, "potencial_alucinacao", consolidated_df[consolidated_df["potential_hallucination"]], ["overall_score", "groundedness_score"], [True, True]))
        interesting_cases.extend(_case_rows(consolidated_df, "retrieval_failure", consolidated_df[consolidated_df["retrieval_failure"]], ["overall_score"], [True]))
        interesting_cases.extend(_case_rows(consolidated_df, "correta_com_recuperacao_ruim", consolidated_df[consolidated_df["correct_with_weak_retrieval"]], ["overall_generation_score", "overall_score"], [False, False]))

        interesting_cases_csv = analysis_dir / "casos_interessantes.csv"
        _write_csv(interesting_cases_csv, interesting_cases)

        figure_paths: list[Path] = []
        if not consolidated_df.empty:
            score_by_model = consolidated_df.groupby("modelo")["overall_score"].mean().reindex(model_order)
            retrieval_by_model = consolidated_df.groupby("modelo")["overall_retrieval_score"].mean().reindex(model_order)
            generation_by_model = consolidated_df.groupby("modelo")["overall_generation_score"].mean().reindex(model_order)
            model_slices = {model: consolidated_df.loc[consolidated_df["modelo"] == model] for model in model_order}
            figure_paths += _make_grouped_bar_chart(
                "Comparacao entre modelos",
                model_order,
                {
                    "overall": score_by_model.fillna(0).tolist(),
                    "retrieval": retrieval_by_model.fillna(0).tolist(),
                    "geracao": generation_by_model.fillna(0).tolist(),
                },
                "Score medio",
                figures_dir,
                "comparacao_modelos",
            )
            figure_paths += _make_boxplot(
                "Distribuicao do tempo de resposta por modelo",
                [model_slices[model]["tempo_resposta_s"].dropna().tolist() for model in model_order],
                model_order,
                "Tempo (s)",
                figures_dir,
                "tempo_resposta",
            )
            figure_paths += _make_boxplot(
                "Consumo de tokens por modelo",
                [model_slices[model]["tokens"].dropna().tolist() for model in model_order],
                model_order,
                "Tokens",
                figures_dir,
                "consumo_tokens",
            )
            figure_paths += _make_histogram(
                "Distribuicao dos tamanhos das respostas",
                consolidated_df["response_length_words"].dropna().tolist(),
                "Palavras por resposta",
                figures_dir,
                "distribuicao_tamanhos_respostas",
            )
            figure_paths += _make_boxplot(
                "Similaridade dos chunks por modelo",
                [model_slices[model]["similaridade_media_chunks"].dropna().tolist() for model in model_order],
                model_order,
                "Similaridade media",
                figures_dir,
                "similaridade_chunks",
            )
            figure_paths += _make_histogram(
                "Distribuicao das similaridades dos chunks",
                consolidated_df["similaridade_media_chunks"].dropna().tolist(),
                "Similaridade media",
                figures_dir,
                "distribuicao_similaridades",
            )
            figure_paths += _make_bar_chart(
                "Ranking final dos modelos",
                model_order,
                score_by_model.fillna(0).tolist(),
                "Overall score medio",
                figures_dir,
                "ranking_final",
            )
            figure_paths += _make_bar_chart(
                "Taxa de retrieval hit por modelo",
                model_order,
                consolidated_df.groupby("modelo")["retrieval_hit"].mean().reindex(model_order).fillna(0).tolist(),
                "Taxa de hit",
                figures_dir,
                "taxa_retrieval_hit",
            )
            figure_paths += _make_bar_chart(
                "Hit Rate@1 por modelo",
                model_order,
                consolidated_df.groupby("modelo")["hit_rate_at_1"].mean().reindex(model_order).fillna(0).tolist(),
                "Hit Rate@1",
                figures_dir,
                "hit_rate_at_1",
            )
            figure_paths += _make_bar_chart(
                "MRR por modelo",
                model_order,
                consolidated_df.groupby("modelo")["mrr"].mean().reindex(model_order).fillna(0).tolist(),
                "MRR",
                figures_dir,
                "mrr",
            )
            figure_paths += _make_bar_chart(
                "Precision@K por modelo",
                model_order,
                consolidated_df.groupby("modelo")["precision_at_k"].mean().reindex(model_order).fillna(0).tolist(),
                "Precision@K",
                figures_dir,
                "precision_at_k",
            )
            figure_paths += _make_bar_chart(
                "Recall@K por modelo",
                model_order,
                consolidated_df.groupby("modelo")["recall_at_k"].mean().reindex(model_order).fillna(0).tolist(),
                "Recall@K",
                figures_dir,
                "recall_at_k",
            )

            k_max = int(max(consolidated_df["top_k"].dropna().max() if "top_k" in consolidated_df.columns else 0, consolidated_df["retrieved_rank"].dropna().max() if consolidated_df["retrieved_rank"].notna().any() else 0, 5))
            k_values = list(range(1, min(k_max, 10) + 1))
            if k_values:
                k_frame_rows = []
                for model in model_order:
                    subset = consolidated_df[consolidated_df["modelo"] == model]
                    ranks = subset["retrieved_rank"].dropna().astype(int).tolist()
                    for k in k_values:
                        hit_rate = sum(rank <= k for rank in ranks) / len(ranks) if ranks else 0.0
                        precision = sum(1 / k for rank in ranks if rank <= k) / len(ranks) if ranks else 0.0
                        recall = hit_rate
                        k_frame_rows.append({"modelo": model, "k": k, "hit_rate": hit_rate, "precision": precision, "recall": recall})

                k_frame = pd.DataFrame(k_frame_rows)
                hit_series = {model: k_frame[k_frame["modelo"] == model].sort_values("k")["hit_rate"].tolist() for model in model_order}
                precision_series = {model: k_frame[k_frame["modelo"] == model].sort_values("k")["precision"].tolist() for model in model_order}
                recall_series = {model: k_frame[k_frame["modelo"] == model].sort_values("k")["recall"].tolist() for model in model_order}

                fig, ax = plt.subplots(figsize=(11, 6))
                for model in model_order:
                    ax.plot(k_values, hit_series.get(model, []), marker="o", linewidth=2, label=model)
                ax.set_title("Hit Rate@K")
                ax.set_xlabel("K")
                ax.set_ylabel("Hit Rate")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                figure_paths += _save_figure(fig, figures_dir, "hit_rate_at_k")

                fig, ax = plt.subplots(figsize=(11, 6))
                for model in model_order:
                    ax.plot(k_values, recall_series.get(model, []), marker="o", linewidth=2, label=model)
                ax.set_title("Recall@K")
                ax.set_xlabel("K")
                ax.set_ylabel("Recall")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                figure_paths += _save_figure(fig, figures_dir, "recall_at_k_curva")

                fig, ax = plt.subplots(figsize=(11, 6))
                for model in model_order:
                    ax.plot(k_values, precision_series.get(model, []), marker="o", linewidth=2, label=model)
                ax.set_title("Precision@K")
                ax.set_xlabel("K")
                ax.set_ylabel("Precision")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                figure_paths += _save_figure(fig, figures_dir, "precision_at_k_curva")

            figure_paths += _make_heatmap(
                "Correlacoes entre metricas numericas",
                consolidated_df[[column for column in NUMERIC_METRICS if column in consolidated_df.columns]].corr(numeric_only=True).fillna(0),
                figures_dir,
                "heatmap_correlacoes",
            )
            figure_paths += _make_scatter(
                "Tempo de resposta x Qualidade",
                consolidated_df["tempo_resposta_s"].fillna(0).tolist(),
                consolidated_df["overall_score"].fillna(0).tolist(),
                "Tempo de resposta (s)",
                "Overall score",
                figures_dir,
                "tempo_x_qualidade",
            )
            figure_paths += _make_scatter(
                "Tokens x Qualidade",
                consolidated_df["tokens"].fillna(0).tolist(),
                consolidated_df["overall_score"].fillna(0).tolist(),
                "Tokens totais",
                "Overall score",
                figures_dir,
                "tokens_x_qualidade",
            )
            figure_paths += _make_radar_chart(
                "Radar chart dos modelos",
                ["Retrieval", "Generation", "Hit@1", "MRR", "Similarity", "Overall"],
                {
                    model: [
                        float(model_slices[model]["overall_retrieval_score"].mean() or 0.0),
                        float(model_slices[model]["overall_generation_score"].mean() or 0.0),
                        float(model_slices[model]["hit_rate_at_1"].mean() or 0.0),
                        float(model_slices[model]["mrr"].mean() or 0.0),
                        float(model_slices[model]["similaridade_media_chunks"].mean() or 0.0),
                        float(model_slices[model]["overall_score"].mean() or 0.0),
                    ]
                    for model in model_order
                },
                figures_dir,
                "radar_modelos",
            )

        model_metrics_table = _markdown_table(
            ["Metrica", "Media", "Desvio padrao", "IC95 inferior", "IC95 superior", "n"],
            [
                [row.metric, f"{row.mean:.4f}", f"{row.std:.4f}", f"{row.ci95_low:.4f}", f"{row.ci95_high:.4f}", str(row.n)]
                for row in summary_rows
                if row.modelo == (model_order[0] if model_order else row.modelo)
            ],
        )

        top_models = ranking_df.sort_values("overall_score", ascending=False).head(5)
        ranking_rows = [[str(row.modelo), f"{row.overall_score:.4f}", f"{row.overall_retrieval_score:.4f}", f"{row.overall_generation_score:.4f}"] for _, row in top_models.iterrows()]

        executive_lines = [
            "## Estatisticas gerais",
            f"- Numero de perguntas: {general_stats['numero_perguntas']}",
            f"- Numero de modelos avaliados: {general_stats['numero_modelos_avaliados']}",
            f"- Tempo medio global de resposta: {general_stats['tempo_medio_global_s']:.3f} s",
            f"- Tokens totais de entrada: {general_stats['tokens_entrada_total']:.0f}",
            f"- Tokens totais de saida: {general_stats['tokens_saida_total']:.0f}",
            f"- Tokens totais: {general_stats['tokens_totais_total']:.0f}",
            f"- Resposta media: {general_stats['resposta_media_chars']:.1f} caracteres e {general_stats['resposta_media_words']:.1f} palavras",
            "",
            "## Ranking geral",
            _markdown_table(["Modelo", "Overall score", "Retrieval score", "Generation score"], ranking_rows),
            "",
            "## Leitura objetiva",
            f"- Melhor modelo na media: {top_models.iloc[0]['modelo']}" if not top_models.empty else "- Melhor modelo na media: indisponivel",
            f"- Recuperacao media: {consolidated_df['overall_retrieval_score'].mean():.4f}",
            f"- Geracao media: {consolidated_df['overall_generation_score'].dropna().mean():.4f}" if consolidated_df['overall_generation_score'].notna().any() else "- Geracao media: indisponivel",
            f"- Taxa de retrieval hit: {consolidated_df['retrieval_hit'].mean():.4f}",
            "",
            "## Casos de interesse",
            f"- Casos de retrieval failure: {int(consolidated_df['retrieval_failure'].sum())}",
            f"- Casos sem contexto recuperado: {int(consolidated_df['no_context_recovered'].sum())}",
            f"- Casos com potencial alucinacao: {int(consolidated_df['potential_hallucination'].sum())}",
            f"- Casos corretos mesmo com recuperacao ruim: {int(consolidated_df['correct_with_weak_retrieval'].sum())}",
        ]
        executive_report_path = analysis_dir / "relatorio_executivo.md"
        _write_markdown_report(executive_report_path, "Relatorio Executivo do Experimento", executive_lines)
        if report_markdown != executive_report_path:
            _write_markdown_report(report_markdown, "Relatorio Executivo do Experimento", executive_lines)

        interesting_cases_df = pd.DataFrame(interesting_cases) if interesting_cases else pd.DataFrame(columns=["case_type"])
        case_counts = {
            "melhores_respostas": int((interesting_cases_df[interesting_cases_df["case_type"] == "melhores_respostas"].shape[0]) if not interesting_cases_df.empty else 0),
            "piores_respostas": int((interesting_cases_df[interesting_cases_df["case_type"] == "piores_respostas"].shape[0]) if not interesting_cases_df.empty else 0),
            "maiores_similaridades": int((interesting_cases_df[interesting_cases_df["case_type"] == "maiores_similaridades"].shape[0]) if not interesting_cases_df.empty else 0),
            "menores_similaridades": int((interesting_cases_df[interesting_cases_df["case_type"] == "menores_similaridades"].shape[0]) if not interesting_cases_df.empty else 0),
            "potencial_alucinacao": int((interesting_cases_df[interesting_cases_df["case_type"] == "potencial_alucinacao"].shape[0]) if not interesting_cases_df.empty else 0),
            "retrieval_failure": int((interesting_cases_df[interesting_cases_df["case_type"] == "retrieval_failure"].shape[0]) if not interesting_cases_df.empty else 0),
            "correta_com_recuperacao_ruim": int((interesting_cases_df[interesting_cases_df["case_type"] == "correta_com_recuperacao_ruim"].shape[0]) if not interesting_cases_df.empty else 0),
        }

        stats_lines = [
            "## Estatisticas gerais",
            _markdown_table(
                ["Indicador", "Valor"],
                [[key.replace("_", " ").capitalize(), f"{value:.4f}" if isinstance(value, float) else str(value)] for key, value in general_stats.items()],
            ),
            "",
            "## Estatisticas descritivas",
            _markdown_table(
                ["Metrica", "Media", "Desvio padrao", "Minimo", "Maximo"],
                [
                    [
                        column,
                        f"{consolidated_df[column].mean():.4f}",
                        f"{consolidated_df[column].std():.4f}",
                        f"{consolidated_df[column].min():.4f}",
                        f"{consolidated_df[column].max():.4f}",
                    ]
                    for column in [col for col in NUMERIC_METRICS if col in consolidated_df.columns]
                ],
            ),
            "",
            "## Matriz consolidada por modelo",
            _markdown_table(
                list(wide_matrix.columns),
                [[str(value) if idx == 0 else f"{float(value):.4f}" if pd.notna(value) else "" for idx, value in enumerate(row)] for row in wide_matrix.itertuples(index=False, name=None)],
            ),
            "",
            "## Ranking geral",
            _markdown_table(
                ["rank", "modelo", "overall_score", "overall_retrieval_score", "overall_generation_score"],
                [
                    [
                        str(row.rank),
                        str(row.modelo),
                        f"{row.overall_score:.4f}" if pd.notna(row.overall_score) else "",
                        f"{row.overall_retrieval_score:.4f}" if pd.notna(row.overall_retrieval_score) else "",
                        f"{row.overall_generation_score:.4f}" if pd.notna(row.overall_generation_score) else "",
                    ]
                    for row in ranking_df.itertuples(index=False)
                ],
            ),
            "",
            "## Casos interessantes",
            _markdown_table(["Categoria", "Quantidade"], [[name, str(count)] for name, count in case_counts.items()]),
            "",
            "## Figura chave",
            *[f"- [{path.name}]({path.relative_to(analysis_dir).as_posix()})" for path in figure_paths[: min(12, len(figure_paths))]],
        ]
        _write_markdown_report(analysis_dir / "analise_estatistica.md", "Analise Estatistica do Experimento", stats_lines)

        discussion_lines = [
            "## Principais achados",
            f"O conjunto analisado possui {general_stats['numero_perguntas']} perguntas avaliadas por {general_stats['numero_modelos_avaliados']} modelos, com media global de resposta de {general_stats['tempo_medio_global_s']:.3f} s.",
            f"O melhor desempenho agregado foi observado em {top_models.iloc[0]['modelo']}" if not top_models.empty else "O melhor desempenho agregado nao pôde ser determinado.",
            f"A taxa media de retrieval hit foi {consolidated_df['retrieval_hit'].mean():.4f}, enquanto a similaridade media dos chunks ficou em {consolidated_df['similaridade_media_chunks'].dropna().mean():.4f}.",
            "",
            "## Interpretacao dos resultados",
            "Modelos com melhor equilibrio entre retrieval e geracao tendem a aparecer no topo do ranking geral, o que sugere que a combinacao de recuperacao semantica e aderencia textual explica boa parte da variacao de qualidade.",
            "Casos com baixa similaridade recuperada, mas com boa pontuacao gerativa, indicam resiliencia do gerador em cenarios com contexto subotimo.",
            "",
            "## Pontos fortes",
            "- O pipeline produz rastreabilidade completa entre pergunta, contexto recuperado e resposta final.",
            "- Ha medidas de retrieval e geracao em nivel fino, o que facilita uma discussão metodologica mais defensavel no TCC.",
            "- Os artefatos gerados permitem reproduzir tabelas e figuras diretamente a partir do CSV bruto.",
            "",
            "## Limitacoes",
            "- As metricas de qualidade da resposta dependem do avaliador por LLM quando habilitado, o que introduz variabilidade adicional.",
            "- O benchmark considera a base local e a configuracao corrente de top_k, portanto os resultados nao generalizam automaticamente para outras bases ou hiperparametros.",
            "- Recuperacao e geracao sao afetadas por escolhas de chunking e embedding que podem favorecer alguns modelos em relacao a outros.",
            "",
            "## Ameacas a validade",
            "- O numero de perguntas por modelo precisa ser suficientemente grande para reduzir a incerteza estatistica.",
            "- A comparacao entre modelos pode ser influenciada pela sensibilidade do juiz automatico e pelo formato da resposta produzida.",
            "- A metricas de retrieval assumem uma unica fonte correta principal por pergunta, o que simplifica a avaliacao mas pode subestimar respostas parcialmente corretas.",
            "",
            "## Melhorias futuras",
            "- Expandir o conjunto de perguntas e incluir mais categorias de dificuldade.",
            "- Testar diferentes valores de top_k, chunking e estrategias de reranking.",
            "- Incorporar avaliacao humana amostral para calibrar a confianca nas notas automaticas.",
            "",
            "## Texto adaptavel para o TCC",
            "Os resultados mostram que a qualidade final do sistema de RAG nao depende apenas do modelo gerador, mas da interacao entre recuperacao, qualidade dos chunks e robustez do modelo diante de contexto parcialmente subotimo. Em especial, os modelos com melhor equilibrio entre alta taxa de retrieval hit, maior MRR e pontuacao gerativa consistente apresentaram o melhor desempenho agregado. Por outro lado, os casos de retrieval failure e de potencial alucinacao evidenciam que uma recuperacao fraca aumenta o risco de respostas menos aderentes ao documento de referencia, reforcando a necessidade de calibrar chunking, top_k e estrategia de avaliacao para garantir confiabilidade experimental.",
        ]
        _write_markdown_report(analysis_dir / "discussao_automatica.md", "Discussao Automatica do Experimento", discussion_lines)

        return {
            "consolidated_csv": consolidated_csv,
            "summary_csv": summary_csv_path,
            "report_markdown": executive_report_path,
            "analysis_markdown": analysis_dir / "analise_estatistica.md",
            "discussion_markdown": analysis_dir / "discussao_automatica.md",
            "ranking_csv": ranking_csv_path,
            "interesting_cases_csv": interesting_cases_csv,
            "results_summary_csv": resultados_resumidos_csv,
            "matrix_csv": matrix_csv_path,
            "figure_paths": figure_paths,
            "ranked_models": list(zip(ranking_df["modelo"].tolist(), ranking_df["overall_score"].fillna(0).tolist())),
            "records": consolidated_df.to_dict(orient="records"),
        }
