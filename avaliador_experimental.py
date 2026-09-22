from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.evaluation import ExperimentEvaluator
from rag_service import create_rag_service


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
    parser = argparse.ArgumentParser(description="Executa o experimento RAG em lote para multiplos modelos do Bedrock.")
    parser.add_argument("--dataset", default="datasets/dataset_padrao_ouro.json", help="Arquivo do dataset padrao-ouro.")
    parser.add_argument("--results", default="results/resultados_experimento.csv", help="CSV bruto de resultados.")
    parser.add_argument("--models", default=None, help="Lista separada por virgulas com os modelos a avaliar.")
    parser.add_argument("--top-k", type=int, default=None, help="Quantidade de chunks recuperados por pergunta.")
    parser.add_argument("--log-file", default="results/avaliador_experimental.log", help="Arquivo opcional de log.")
    args = parser.parse_args()

    config = ExperimentConfig.from_env()
    model_ids = tuple(item.strip() for item in args.models.split(",") if item.strip()) if args.models else config.model_ids
    effective_top_k = args.top_k if args.top_k is not None else config.top_k
    log_path = Path(args.log_file) if args.log_file else None
    _configure_logging(log_path)
    logger = logging.getLogger("avaliador_experimental")

    dataset_path = Path(args.dataset)
    results_path = Path(args.results)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rag_service = create_rag_service()
    evaluator = ExperimentEvaluator(
        rag_service=rag_service,
        model_ids=model_ids,
        top_k=effective_top_k,
        retrieval_variants=config.retrieval_variants(),
        logger_=logger,
    )

    logger.info("Iniciando avaliacao de %s perguntas com %s modelos.", len(dataset), len(model_ids))
    output = evaluator.run(dataset=dataset, output_csv=results_path)
    logger.info("Resultados salvos em %s (%s registros).", output.raw_results_csv, len(output.records))


if __name__ == "__main__":
    main()
