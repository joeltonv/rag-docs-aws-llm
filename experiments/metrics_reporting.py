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
import numpy as np
import pandas as pd

from experiments.analysis_experiment import JudgeServiceProtocol, _canonical_path_key, _judge_record, _load_dataset, _load_rows, _parse_json_field


logger = logging.getLogger(__name__)

RETRIEVAL_METRICS = ["hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", "similaridade_media_chunks"]
GENERATION_METRICS = [
	"groundedness_score",
	"corretude_score",
	"completude_score",
	"clareza_score",
	"precisao_tecnica_score",
	"alucinacao_score",
]


@dataclass(frozen=True)
class SummaryRow:
	grupo: str
	variante: str
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


def _serialize_json(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		path.write_text("", encoding="utf-8")
		return

	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
	header_line = "| " + " | ".join(headers) + " |"
	separator_line = "| " + " | ".join("---" for _ in headers) + " |"
	body_lines = ["| " + " | ".join(row) + " |" for row in rows]
	return "\n".join([header_line, separator_line, *body_lines])


def _write_markdown_report(output_path: Path, title: str, sections: list[str]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join([f"# {title}", "", *sections]).rstrip() + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> list[Path]:
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


def _load_result_rows(results_csv: Path) -> pd.DataFrame:
	results_df = _load_rows(results_csv)
	if "variacao_retrieval" not in results_df.columns:
		results_df = results_df.copy()
		results_df["variacao_retrieval"] = "production"
	return results_df


def _build_retrieval_metrics(row: pd.Series, dataset_row: pd.Series) -> dict[str, Any]:
	retrieved_results = _parse_json_field(row.get("chunks_recuperados_json"), [])
	source_paths = _parse_json_field(row.get("source_paths_chunks"), [])
	if not isinstance(source_paths, list):
		source_paths = []

	similarity_values = [value for value in _parse_json_field(row.get("similaridades_chunks"), []) if isinstance(value, (int, float))]
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

	top_k = int(_ensure_float(row.get("top_k")) or len(retrieved_results) or 1)
	top_k = max(top_k, 1)
	hit_rate_at_k = 1.0 if relevant_rank is not None and relevant_rank <= top_k else 0.0
	recall_at_k = float(relevant_count > 0)
	precision_at_k = relevant_count / max(top_k, 1)
	mrr = 1.0 / relevant_rank if relevant_rank is not None else 0.0
	similarity_mean = statistics.mean(similarity_values) if similarity_values else _ensure_float(row.get("similaridade_media_chunks"))

	return {
		"retrieval_hit": bool(relevant_rank),
		"retrieved_rank": relevant_rank,
		"hit_rate_at_k": hit_rate_at_k,
		"recall_at_k": recall_at_k,
		"precision_at_k": precision_at_k,
		"mrr": mrr,
		"similaridade_media_chunks": similarity_mean,
		"source_paths_chunks": json.dumps(source_paths, ensure_ascii=False),
	}


def _normalized_retrieval_score(row: pd.Series) -> float | None:
	scores = []
	for metric in RETRIEVAL_METRICS:
		value = _ensure_float(row.get(metric))
		if value is not None:
			scores.append(max(0.0, min(1.0, value)))
	return statistics.mean(scores) if scores else None


def _normalized_generation_score(row: pd.Series) -> float | None:
	scores = []
	for metric in GENERATION_METRICS:
		value = _ensure_float(row.get(metric))
		if value is not None:
			scores.append(value / 5.0)
	return statistics.mean(scores) if scores else None


def _row_composite_score(row: pd.Series) -> float | None:
	retrieval_score = _normalized_retrieval_score(row)
	generation_score = _normalized_generation_score(row)
	scores = [value for value in (retrieval_score, generation_score) if value is not None]
	return statistics.mean(scores) if scores else None


def _group_summary(rows: pd.DataFrame, group_columns: list[str], metrics: list[str], group_name: str) -> list[SummaryRow]:
	summary_rows: list[SummaryRow] = []
	grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)

	for _, row in rows.iterrows():
		variant = str(row.get("variacao_retrieval", "production"))
		model = str(row.get("modelo", ""))
		for metric in metrics:
			value = _ensure_float(row.get(metric))
			if value is not None:
				grouped[(variant, model, metric)].append(value)

	variants = list(dict.fromkeys(rows["variacao_retrieval"].fillna("production").astype(str).tolist()))
	models = list(dict.fromkeys(rows["modelo"].fillna("").astype(str).tolist()))

	for variant in variants:
		for model in models:
			for metric in metrics:
				values = grouped.get((variant, model, metric), [])
				mean_value, low_value, high_value, std_value = _mean_confidence_interval(values)
				summary_rows.append(
					SummaryRow(
						grupo=group_name,
						variante=variant,
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


def _aggregate_variant_summary(rows: pd.DataFrame, metrics: list[str], group_name: str) -> list[dict[str, Any]]:
	summary_records: list[dict[str, Any]] = []
	grouped = rows.groupby("variacao_retrieval", dropna=False)
	for variant, subset in grouped:
		record: dict[str, Any] = {"grupo": group_name, "variacao_retrieval": str(variant)}
		for metric in metrics:
			values = [_ensure_float(value) for value in subset[metric].tolist() if _ensure_float(value) is not None]
			mean_value, low_value, high_value, std_value = _mean_confidence_interval([float(value) for value in values if value is not None])
			record[f"{metric}_mean"] = mean_value
			record[f"{metric}_ci95_low"] = low_value
			record[f"{metric}_ci95_high"] = high_value
			record[f"{metric}_std"] = std_value
			record[f"{metric}_n"] = len(values)

		summary_records.append(record)

	return summary_records


def _baseline_delta_table(summary_df: pd.DataFrame, metrics: list[str], baseline_variant: str) -> list[dict[str, Any]]:
	if summary_df.empty or baseline_variant not in set(summary_df["variacao_retrieval"].astype(str)):
		return []

	baseline_row = summary_df[summary_df["variacao_retrieval"].astype(str) == baseline_variant].iloc[0]
	rows: list[dict[str, Any]] = []
	for _, row in summary_df.iterrows():
		current_variant = str(row["variacao_retrieval"])
		for metric in metrics:
			current_value = _ensure_float(row.get(f"{metric}_mean"))
			baseline_value = _ensure_float(baseline_row.get(f"{metric}_mean"))
			if current_value is None or baseline_value is None:
				continue
			absolute_delta = current_value - baseline_value
			relative_gain = (absolute_delta / baseline_value * 100.0) if baseline_value != 0 else None
			rows.append(
				{
					"variacao_retrieval": current_variant,
					"metric": metric,
					"mean": current_value,
					"baseline": baseline_value,
					"delta_absoluto": absolute_delta,
					"ganho_percentual": relative_gain,
				}
			)

	return rows


class ExperimentMetricsPipeline:
	def __init__(self, judge_service: JudgeServiceProtocol | None, judge_model_id: str, judge_system_prompt: str, logger_: logging.Logger | None = None) -> None:
		self.judge_service = judge_service
		self.judge_model_id = judge_model_id
		self.judge_system_prompt = judge_system_prompt
		self.logger = logger_ or logger

	def run(self, dataset_path: Path, results_csv: Path, consolidated_csv: Path, report_markdown: Path, figures_dir: Path) -> dict[str, Any]:
		dataset_df = _load_dataset(dataset_path)
		results_df = _load_result_rows(results_csv)
		dataset_lookup = dataset_df.set_index("id")

		consolidated_rows: list[dict[str, Any]] = []
		judge_disabled_due_to_error = False

		for _, row in results_df.iterrows():
			example_lookup = dataset_lookup.loc[str(row.get("id"))] if str(row.get("id")) in dataset_lookup.index else None
			if example_lookup is None:
				self.logger.warning("Exemplo %s nao encontrado no dataset; linha ignorada.", row.get("id"))
				continue

			example = example_lookup.iloc[0] if isinstance(example_lookup, pd.DataFrame) else example_lookup
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
					for metric in GENERATION_METRICS:
						consolidated[metric] = None
						consolidated[f"{metric.replace('_score', '')}_comment"] = ""
					consolidated["judge_model_id"] = ""
					consolidated["judge_raw_answer"] = ""
			else:
				for metric in GENERATION_METRICS:
					consolidated[metric] = None
					consolidated[f"{metric.replace('_score', '')}_comment"] = ""
				consolidated["judge_model_id"] = ""
				consolidated["judge_raw_answer"] = ""

			consolidated["retrieval_quality_score"] = _normalized_retrieval_score(pd.Series(consolidated))
			consolidated["generation_quality_score"] = _normalized_generation_score(pd.Series(consolidated))
			consolidated["composite_quality_score"] = _row_composite_score(pd.Series(consolidated))
			consolidated_rows.append(consolidated)

		consolidated_df = pd.DataFrame(consolidated_rows)
		if consolidated_df.empty:
			raise ValueError("Nao ha linhas validas para consolidar.")

		for column in ["tempo_resposta_s", "input_tokens", "output_tokens", "tokens", "answer_length", "question_length", "chunks_recuperados", "similaridade_media_chunks", "retrieved_rank", "hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", *RETRIEVAL_METRICS, *GENERATION_METRICS, "retrieval_quality_score", "generation_quality_score", "composite_quality_score"]:
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
		if consolidated_df[GENERATION_METRICS].notna().any().any():
			consolidated_df["potential_hallucination"] = (
				consolidated_df["groundedness_score"].fillna(5).astype(float).le(2)
				| consolidated_df["alucinacao_score"].fillna(5).astype(float).le(2)
				| consolidated_df["corretude_score"].fillna(5).astype(float).le(2)
			)
		consolidated_df["correct_with_weak_retrieval"] = (
			consolidated_df["retrieval_hit"].fillna(False).astype(bool).eq(False)
			& consolidated_df["generation_quality_score"].fillna(0).astype(float).ge(0.8)
		)
		consolidated_df["response_words_per_token"] = consolidated_df["response_length_words"].where(consolidated_df["tokens"].fillna(0).gt(0)) / consolidated_df["tokens"].replace(0, np.nan)

		consolidated_df.to_csv(consolidated_csv, index=False, encoding="utf-8")

		output_dir = report_markdown.parent
		output_dir.mkdir(parents=True, exist_ok=True)
		figures_dir.mkdir(parents=True, exist_ok=True)

		variant_order = list(dict.fromkeys(consolidated_df["variacao_retrieval"].fillna("production").astype(str).tolist()))
		model_order = list(dict.fromkeys(consolidated_df["modelo"].fillna("").astype(str).tolist()))
		if not model_order:
			model_order = ["desconhecido"]

		retrieval_summary_rows = _group_summary(consolidated_df, ["variacao_retrieval", "modelo"], RETRIEVAL_METRICS, "retrieval")
		generation_summary_rows = _group_summary(consolidated_df, ["variacao_retrieval", "modelo"], GENERATION_METRICS, "generation")

		retrieval_summary_csv = output_dir / "resumo_metricas_retrieval.csv"
		generation_summary_csv = output_dir / "resumo_metricas_generation.csv"
		legacy_summary_csv = output_dir / "resumo_metricas.csv"
		comparative_csv = output_dir / "comparativo_variantes.csv"

		_write_csv(
			retrieval_summary_csv,
			[
				{
					"grupo": row.grupo,
					"variacao_retrieval": row.variante,
					"modelo": row.modelo,
					"metric": row.metric,
					"mean": row.mean,
					"ci95_low": row.ci95_low,
					"ci95_high": row.ci95_high,
					"std": row.std,
					"n": row.n,
				}
				for row in retrieval_summary_rows
			],
		)
		_write_csv(
			legacy_summary_csv,
			[
				{
					"grupo": row.grupo,
					"variacao_retrieval": row.variante,
					"modelo": row.modelo,
					"metric": row.metric,
					"mean": row.mean,
					"ci95_low": row.ci95_low,
					"ci95_high": row.ci95_high,
					"std": row.std,
					"n": row.n,
				}
				for row in retrieval_summary_rows
			],
		)
		_write_csv(
			generation_summary_csv,
			[
				{
					"grupo": row.grupo,
					"variacao_retrieval": row.variante,
					"modelo": row.modelo,
					"metric": row.metric,
					"mean": row.mean,
					"ci95_low": row.ci95_low,
					"ci95_high": row.ci95_high,
					"std": row.std,
					"n": row.n,
				}
				for row in generation_summary_rows
			],
		)

		retrieval_variant_summary = _aggregate_variant_summary(consolidated_df, RETRIEVAL_METRICS, "retrieval")
		generation_variant_summary = _aggregate_variant_summary(consolidated_df, GENERATION_METRICS, "generation")
		retrieval_variant_df = pd.DataFrame(retrieval_variant_summary)
		generation_variant_df = pd.DataFrame(generation_variant_summary)

		baseline_variant = "baseline" if "baseline" in set(retrieval_variant_df.get("variacao_retrieval", pd.Series(dtype=str)).astype(str).tolist()) else (variant_order[0] if variant_order else "production")
		retrieval_delta_rows = _baseline_delta_table(retrieval_variant_df, RETRIEVAL_METRICS, baseline_variant)
		generation_delta_rows = _baseline_delta_table(generation_variant_df, GENERATION_METRICS, baseline_variant)
		comparative_rows = retrieval_delta_rows + generation_delta_rows
		_write_csv(comparative_csv, comparative_rows)

		model_ranking_df = consolidated_df.groupby("modelo", dropna=False).agg(
			retrieval_quality_score=("retrieval_quality_score", "mean"),
			generation_quality_score=("generation_quality_score", "mean"),
			composite_quality_score=("composite_quality_score", "mean"),
		).reset_index()
		model_ranking_df = model_ranking_df.sort_values(["composite_quality_score", "retrieval_quality_score", "generation_quality_score"], ascending=False)
		model_ranking_df["rank"] = range(1, len(model_ranking_df) + 1)

		# figures
		if not retrieval_variant_df.empty:
			_make_bar_chart(
				"Retrieval médio por variante",
				retrieval_variant_df["variacao_retrieval"].astype(str).tolist(),
				retrieval_variant_df[[f"{metric}_mean" for metric in RETRIEVAL_METRICS]].mean(axis=1).fillna(0).tolist(),
				"Score médio",
				figures_dir,
				"retrieval_medio_variantes",
			)
		if not generation_variant_df.empty:
			_make_bar_chart(
				"Generation médio por variante",
				generation_variant_df["variacao_retrieval"].astype(str).tolist(),
				generation_variant_df[[f"{metric}_mean" for metric in GENERATION_METRICS]].mean(axis=1).fillna(0).tolist(),
				"Score médio",
				figures_dir,
				"generation_medio_variantes",
			)
		if not model_ranking_df.empty:
			_make_bar_chart(
				"Ranking composto por modelo",
				model_ranking_df["modelo"].astype(str).tolist(),
				model_ranking_df["composite_quality_score"].fillna(0).tolist(),
				"Score composto",
				figures_dir,
				"ranking_modelos_composto",
			)

		retrieval_report_lines = [
			"## Métricas de retrieval",
			_markdown_table(
				["Variante", "Modelo", *RETRIEVAL_METRICS],
				[
					[
						row["variacao_retrieval"],
						str(row.get("modelo", "todos")),
						*[f"{row[f'{metric}_mean']:.4f}" for metric in RETRIEVAL_METRICS],
					]
					for _, row in retrieval_variant_df.iterrows()
				],
			) if not retrieval_variant_df.empty else "Sem dados de retrieval.",
			"",
			"## Leitura objetiva",
			f"- Baseline: {baseline_variant}",
			f"- Maior Hit Rate@K: {consolidated_df.groupby('variacao_retrieval')['hit_rate_at_k'].mean().idxmax() if not consolidated_df.empty else 'indisponivel'}",
			f"- Maior MRR: {consolidated_df.groupby('variacao_retrieval')['mrr'].mean().idxmax() if not consolidated_df.empty else 'indisponivel'}",
		]

		generation_report_lines = [
			"## Métricas de generation",
			_markdown_table(
				["Variante", "Modelo", *GENERATION_METRICS],
				[
					[
						row["variacao_retrieval"],
						str(row.get("modelo", "todos")),
						*[f"{row[f'{metric}_mean']:.4f}" for metric in GENERATION_METRICS],
					]
					for _, row in generation_variant_df.iterrows()
				],
			) if not generation_variant_df.empty else "Sem dados de generation.",
			"",
			"## Leitura objetiva",
			f"- Baseline: {baseline_variant}",
			f"- Melhor groundedness médio: {consolidated_df.groupby('variacao_retrieval')['groundedness_score'].mean().idxmax() if consolidated_df['groundedness_score'].notna().any() else 'indisponivel'}",
			f"- Menor taxa de alucinação: {consolidated_df.groupby('variacao_retrieval')['alucinacao_score'].mean().idxmax() if consolidated_df['alucinacao_score'].notna().any() else 'indisponivel'}",
		]

		comparative_lines = [
			"## Ganho estimado vs baseline",
			_markdown_table(
				["Variante", "Métrica", "Média", "Baseline", "Delta absoluto", "Ganho %"],
				[
					[
						str(row["variacao_retrieval"]),
						str(row["metric"]),
						f"{row['mean']:.4f}",
						f"{row['baseline']:.4f}",
						f"{row['delta_absoluto']:.4f}",
						f"{row['ganho_percentual']:.2f}%" if row["ganho_percentual"] is not None else "n/a",
					]
					for row in comparative_rows
				],
			) if comparative_rows else "Sem comparativo disponível.",
			"",
			"## Ranking composto por modelo",
			_markdown_table(
				["Rank", "Modelo", "Retrieval", "Generation", "Composto"],
				[
					[
						str(int(row["rank"])),
						str(row["modelo"]),
						f"{row['retrieval_quality_score']:.4f}",
						f"{row['generation_quality_score']:.4f}",
						f"{row['composite_quality_score']:.4f}",
					]
					for _, row in model_ranking_df.iterrows()
				],
			) if not model_ranking_df.empty else "Sem ranking disponível.",
			"",
			"## Leitura objetiva",
			f"- Baseline usada: {baseline_variant}",
			f"- Variantes avaliadas: {', '.join(variant_order)}",
			f"- Ganho médio de retrieval na melhor variante: {((retrieval_variant_df[[f'{metric}_mean' for metric in RETRIEVAL_METRICS]].mean(axis=1).max() - retrieval_variant_df[[f'{metric}_mean' for metric in RETRIEVAL_METRICS]].mean(axis=1).min()) if not retrieval_variant_df.empty else 0.0):.4f}",
		]

		retrieval_report_path = output_dir / "relatorio_retrieval.md"
		generation_report_path = output_dir / "relatorio_generation.md"
		comparative_report_path = output_dir / "relatorio_comparativo.md"

		_write_markdown_report(retrieval_report_path, "Relatorio de Retrieval do Experimento", retrieval_report_lines)
		_write_markdown_report(generation_report_path, "Relatorio de Generation do Experimento", generation_report_lines)
		_write_markdown_report(comparative_report_path, "Relatorio Comparativo do Experimento", comparative_lines)
		if report_markdown != comparative_report_path:
			_write_markdown_report(report_markdown, "Relatorio Comparativo do Experimento", comparative_lines)

		top_models = model_ranking_df.sort_values("composite_quality_score", ascending=False).head(5)

		return {
			"report_markdown": comparative_report_path,
			"analysis_markdown": retrieval_report_path,
			"discussion_markdown": generation_report_path,
			"retrieval_report_markdown": retrieval_report_path,
			"generation_report_markdown": generation_report_path,
			"comparison_report_markdown": comparative_report_path,
			"summary_csv": retrieval_summary_csv,
			"generation_summary_csv": generation_summary_csv,
			"comparative_csv": comparative_csv,
			"ranked_models": list(zip(top_models["modelo"].tolist(), top_models["composite_quality_score"].fillna(0).tolist())),
			"records": consolidated_df.to_dict(orient="records"),
		}


__all__ = ["ExperimentMetricsPipeline", "SummaryRow"]
