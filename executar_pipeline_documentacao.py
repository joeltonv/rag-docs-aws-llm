from __future__ import annotations

import argparse
import time
from pathlib import Path

from converter_documentacao import convert_documentation_tree
from gerar_embeddings import popular_banco_vetorial
from pipeline_ingestao_rag import pipeline_ingestao_rag


def executar_pipeline_documentacao(
    source_root: str,
    markdown_output: str,
    db_path: str,
    chroma_dir: str,
    convert_workers: int | None = None,
    max_chunk_chars: int | None = None,
    chunk_overlap_sentences: int | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int | None = None,
) -> None:
    source_path = Path(source_root)
    markdown_root = Path(markdown_output)

    conversion_start = time.perf_counter()
    conversion_results = convert_documentation_tree(source_path, markdown_root, max_workers=convert_workers)
    conversion_elapsed = time.perf_counter() - conversion_start
    print(f"[CONVERSÃO] Arquivos convertidos para Markdown: {len(conversion_results)} em {conversion_elapsed:.2f}s")

    ingestion_start = time.perf_counter()
    pipeline_ingestao_rag(
        str(markdown_root),
        db_path=db_path,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_sentences=chunk_overlap_sentences,
    )
    ingestion_elapsed = time.perf_counter() - ingestion_start
    print(f"[INGESTÃO] Markdown persistido em SQLite em {ingestion_elapsed:.2f}s")

    vectorization_start = time.perf_counter()
    popular_banco_vetorial(
        sqlite_path=db_path,
        chroma_dir=chroma_dir,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )
    vectorization_elapsed = time.perf_counter() - vectorization_start
    print(f"[VETORIZAÇÃO] Etapa concluída em {vectorization_elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa o fluxo completo HTML→Markdown→SQLite→Embeddings→ChromaDB."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="docs",
        help="Pasta raiz com a documentação extraída do CHM.",
    )
    parser.add_argument(
        "--markdown-output",
        default="docs_markdown",
        help="Pasta de saída para os arquivos Markdown convertidos.",
    )
    parser.add_argument(
        "--db",
        default="knowledge_base.db",
        help="Caminho do banco SQLite de saída.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="chroma_db",
        help="Diretório persistente do ChromaDB.",
    )
    parser.add_argument(
        "--convert-workers",
        type=int,
        default=None,
        help="Número de processos usados na conversão HTML para Markdown.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Limite máximo de caracteres por chunk; chunks maiores são divididos em subchunks.",
    )
    parser.add_argument(
        "--chunk-overlap-sentences",
        type=int,
        default=None,
        help="Número de sentenças reaproveitadas entre subchunks.",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device do modelo de embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Tamanho do lote usado na vetorização.",
    )

    args = parser.parse_args()
    executar_pipeline_documentacao(
        args.source,
        args.markdown_output,
        args.db,
        args.chroma_dir,
        convert_workers=args.convert_workers,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap_sentences=args.chunk_overlap_sentences,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
    )
