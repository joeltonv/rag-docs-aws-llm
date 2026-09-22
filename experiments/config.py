from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_IDS = (
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "deepseek.v3.2",
    "mistral.mistral-large-3-675b-instruct",
)
DEFAULT_JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-5"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if value is not None else default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um inteiro.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on", "sim"}


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    top_k: int
    translate_query: bool
    description: str


def _parse_model_ids(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None or not raw_value.strip():
        return DEFAULT_MODEL_IDS

    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(values) if values else DEFAULT_MODEL_IDS


@dataclass(frozen=True)
class DatasetBuilderConfig:
    docs_root: Path = Path("docs_markdown")
    output_path: Path = Path("datasets/dataset_padrao_ouro.json")
    target_size: int = 50
    seed: int = 42

    @classmethod
    def from_env(cls) -> "DatasetBuilderConfig":
        return cls(
            docs_root=Path(_env_str("EXPERIMENT_DOCS_ROOT", "docs_markdown")),
            output_path=Path(_env_str("EXPERIMENT_DATASET_OUTPUT", "datasets/dataset_padrao_ouro.json")),
            target_size=_env_int("EXPERIMENT_TARGET_QUESTIONS", 50),
            seed=_env_int("EXPERIMENT_RANDOM_SEED", 42),
        )


@dataclass(frozen=True)
class ExperimentPaths:
    results_dir: Path = Path("results")
    raw_results_csv: Path = Path("results/resultados_experimento.csv")
    consolidated_results_csv: Path = Path("results/resultados_consolidados.csv")
    report_markdown: Path = Path("results/relatorio_experimento.md")
    figures_dir: Path = Path("results/figures")

    @classmethod
    def from_env(cls) -> "ExperimentPaths":
        return cls(
            results_dir=Path(_env_str("EXPERIMENT_RESULTS_DIR", "results")),
            raw_results_csv=Path(_env_str("EXPERIMENT_RESULTS_CSV", "results/resultados_experimento.csv")),
            consolidated_results_csv=Path(_env_str("EXPERIMENT_CONSOLIDATED_CSV", "results/resultados_consolidados.csv")),
            report_markdown=Path(_env_str("EXPERIMENT_REPORT_MD", "results/relatorio_experimento.md")),
            figures_dir=Path(_env_str("EXPERIMENT_FIGURES_DIR", "results/figures")),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetBuilderConfig = DatasetBuilderConfig()
    paths: ExperimentPaths = ExperimentPaths()
    model_ids: tuple[str, ...] = DEFAULT_MODEL_IDS
    top_k: int = 10
    baseline_top_k: int = 4
    compare_variants: bool = True
    include_translation_variants: bool = True
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID
    judge_enabled: bool = True
    judge_system_prompt: str = (
        "Você é um juiz imparcial e rigoroso de respostas de RAG. "
        "Avalie com estrita objetividade, ignorando estilo literário, fluidez ou formatação. "
        "Concentre-se apenas na exatidão técnica, na fidelidade ao contexto e na completude. "
        "Responda somente em JSON válido e sem comentários fora da estrutura solicitada."
    )

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        dataset = DatasetBuilderConfig.from_env()
        paths = ExperimentPaths.from_env()
        return cls(
            dataset=dataset,
            paths=paths,
            model_ids=_parse_model_ids(os.getenv("EXPERIMENT_MODEL_IDS")),
            top_k=_env_int("EXPERIMENT_TOP_K", 10),
            baseline_top_k=_env_int("EXPERIMENT_BASELINE_TOP_K", 4),
            compare_variants=_env_bool("EXPERIMENT_COMPARE_VARIANTS", True),
            include_translation_variants=_env_bool("EXPERIMENT_INCLUDE_TRANSLATION_VARIANTS", True),
            judge_model_id=_env_str("EXPERIMENT_JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL_ID),
            judge_enabled=_env_bool("EXPERIMENT_ENABLE_JUDGE", True),
            judge_system_prompt=_env_str(
                "EXPERIMENT_JUDGE_SYSTEM_PROMPT",
                "Você é um juiz imparcial e rigoroso de respostas de RAG. Avalie com estrita objetividade, ignorando estilo literário, fluidez ou formatação. Concentre-se apenas na exatidão técnica, na fidelidade ao contexto e na completude. Responda somente em JSON válido e sem comentários fora da estrutura solicitada.",
            ),
        )

    def retrieval_variants(self) -> tuple[RetrievalVariant, ...]:
        if not self.compare_variants:
            return (
                RetrievalVariant(
                    name="production",
                    top_k=self.top_k,
                    translate_query=True,
                    description="Configuração principal definida pelo ambiente.",
                ),
            )

        variants = [
            RetrievalVariant(
                name="baseline",
                top_k=self.baseline_top_k,
                translate_query=False,
                description="Linha de base sem tradução e com Top-K reduzido.",
            ),
            RetrievalVariant(
                name="topk_aumentado",
                top_k=self.top_k,
                translate_query=False,
                description="Top-K ampliado sem tradução.",
            ),
        ]

        if self.include_translation_variants:
            variants.insert(
                1,
                RetrievalVariant(
                    name="baseline_traduzido",
                    top_k=self.baseline_top_k,
                    translate_query=True,
                    description="Linha de base com tradução de consulta.",
                ),
            )
            variants.append(
                RetrievalVariant(
                    name="topk_aumentado_traduzido",
                    top_k=self.top_k,
                    translate_query=True,
                    description="Top-K ampliado com tradução de consulta.",
                )
            )

        return tuple(variants)
