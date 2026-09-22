from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bedrock_service import BedrockService
from experiments.config import ExperimentConfig
from experiments.metrics_reporting import ExperimentMetricsPipeline


def _configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    base_output_dir = Path("results/analise_experimento")
    parser = argparse.ArgumentParser(description="Calcula as metricas do experimento e gera os relatorios finais.")
    parser.add_argument("--dataset", default="datasets/dataset_padrao_ouro.json", help="Dataset padrao-ouro em JSON.")
    parser.add_argument("--results", default="results/resultados_experimento.csv", help="CSV bruto gerado pelo avaliador.")
    parser.add_argument("--consolidated", default=str(base_output_dir / "resultados_consolidados.csv"), help="CSV consolidado de metricas.")
    parser.add_argument("--report", default=str(base_output_dir / "relatorio_executivo.md"), help="Relatorio final em Markdown.")
    parser.add_argument("--figures-dir", default=str(base_output_dir / "figures"), help="Diretorio dos graficos gerados.")
    parser.add_argument("--disable-judge", action="store_true", help="Desabilita a avaliacao por LLM-as-a-Judge.")
    parser.add_argument("--log-file", default=str(base_output_dir / "metrics_pipeline.log"), help="Arquivo opcional de log.")
    args = parser.parse_args()

    config = ExperimentConfig.from_env()
    _configure_logging(Path(args.log_file) if args.log_file else None)
    logger = logging.getLogger("metrics_pipeline")

    judge_service = None if args.disable_judge or not config.judge_enabled else BedrockService()
    pipeline = ExperimentMetricsPipeline(
        judge_service=judge_service,
        judge_model_id=config.judge_model_id,
        judge_system_prompt=config.judge_system_prompt,
        logger_=logger,
    )

    logger.info("Calculando metricas com dataset=%s e results=%s.", args.dataset, args.results)
    output = pipeline.run(
        dataset_path=Path(args.dataset),
        results_csv=Path(args.results),
        consolidated_csv=Path(args.consolidated),
        report_markdown=Path(args.report),
        figures_dir=Path(args.figures_dir),
    )
    logger.info("Relatorio gerado em %s.", output["report_markdown"])


if __name__ == "__main__":
    main()
