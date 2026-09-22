from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bedrock_service import BedrockService
from experiments.config import DatasetBuilderConfig
from experiments.dataset_generation import build_dataset, save_dataset


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o dataset padrao-ouro a partir dos Markdown em docs_markdown/")
    parser.add_argument("--docs-root", default="docs_markdown", help="Diretorio raiz com os Markdown convertidos.")
    parser.add_argument("--output", default="datasets/dataset_padrao_ouro.json", help="Arquivo JSON de saida.")
    parser.add_argument("--target-size", type=int, default=50, help="Quantidade aproximada de perguntas a gerar.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para selecao deterministica.")
    parser.add_argument("--use-llm-questions", action="store_true", help="Gera perguntas com Bedrock antes do fallback deterministico.")
    parser.add_argument("--question-model-id", default="us.anthropic.claude-sonnet-5", help="Modelo Bedrock usado para gerar perguntas.")
    parser.add_argument("--question-system-prompt", default="Você é um gerador de perguntas para avaliação científica. Produza perguntas variadas, realistas e semanticamente desafiadoras.", help="Prompt de sistema usado na geração das perguntas.")
    args = parser.parse_args()

    _configure_logging()
    logger = logging.getLogger("dataset_builder")

    config = DatasetBuilderConfig(
        docs_root=Path(args.docs_root),
        output_path=Path(args.output),
        target_size=args.target_size,
        seed=args.seed,
    )

    question_service = BedrockService() if args.use_llm_questions else None

    logger.info("Gerando dataset padrao-ouro em %s a partir de %s.", config.output_path, config.docs_root)
    dataset = build_dataset(
        config.docs_root,
        target_size=config.target_size,
        seed=config.seed,
        question_service=question_service,
        question_model_id=args.question_model_id,
        question_system_prompt=args.question_system_prompt,
    )
    save_dataset(dataset, config.output_path)
    logger.info("Dataset gerado com %s registros.", len(dataset))


if __name__ == "__main__":
    main()
