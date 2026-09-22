import hashlib
import os
import sqlite3
import time
from typing import Any, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Metadata


DEFAULT_EMBEDDING_MODEL_PROVIDER = 'sentence-transformers'
LOCAL_BGE_M3_DIRNAME = 'bge-m3'
LOCAL_BGE_M3_DIMENSION = 1024
BATCH_SIZE = 500


def _resolve_embedding_device(device: str | None) -> str:
    effective_device = (device or os.getenv('EMBEDDING_DEVICE', 'cpu')).strip().lower()
    if effective_device not in {'cpu', 'cuda'}:
        raise ValueError("EMBEDDING_DEVICE deve ser 'cpu' ou 'cuda'.")

    if effective_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada para embeddings, mas não está disponível neste ambiente.")

    return effective_device


def _resolve_batch_size(batch_size: int | None) -> int:
    if batch_size is not None:
        effective_batch_size = batch_size
    else:
        batch_size_value = os.getenv('EMBEDDING_BATCH_SIZE', str(BATCH_SIZE))
        try:
            effective_batch_size = int(batch_size_value)
        except ValueError as exc:
            raise ValueError("EMBEDDING_BATCH_SIZE deve ser um inteiro positivo.") from exc

    if effective_batch_size <= 0:
        raise ValueError('batch_size deve ser maior que zero.')

    return effective_batch_size


def _coerce_embeddings_to_array(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "tolist"):
        raw_embeddings = embeddings.tolist()
    else:
        raw_embeddings = embeddings

    return np.asarray([[float(value) for value in embedding] for embedding in raw_embeddings], dtype=np.float32)


def _resolve_embedding_model() -> tuple[str, str, int, str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(repo_root, 'modelos', LOCAL_BGE_M3_DIRNAME)

    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(
            f"O modelo local BGE-M3 não foi encontrado em '{local_model_path}'."
        )

    return local_model_path, DEFAULT_EMBEDDING_MODEL_PROVIDER, LOCAL_BGE_M3_DIMENSION, 'local'


def _build_embedding_model_metadata(model_name: str, provider: str, dimension: int, source: str) -> dict[str, Any]:
    return {
        'embedding_model': model_name,
        'embedding_model_provider': provider,
        'embedding_model_dimension': dimension,
        'embedding_model_source': source,
    }


def _load_embedding_model(device: str = 'cpu') -> tuple[SentenceTransformer, str, str, int, str]:
    resolved_model_name, provider, dimension, source = _resolve_embedding_model()
    device = _resolve_embedding_device(device)
    try:
        return SentenceTransformer(resolved_model_name, device=device), resolved_model_name, provider, dimension, source
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Falha ao carregar o modelo de embeddings '{resolved_model_name}' no device '{device}'. "
            "O pipeline foi interrompido para preservar a integridade do índice vetorial."
        ) from exc


def preparar_documentos_vetoriais(rows, embedding_backend, embedding_model_metadata: dict[str, Any]):
    documents: List[str] = []
    metadatas: List[Metadata] = []
    ids: List[str] = []

    for row in rows:
        (
            sqlite_id,
            nome_software,
            source_path,
            chunk_ordem,
            subchunk_ordem,
            subchunk_total,
            hierarquia,
            titulo,
            conteudo_texto,
            arquivo_hash,
            arquivo_ingestao_em,
            conteudo_hash,
            conteudo_ingestao_em,
        ) = row

        texto_enriquecido = f"Software: {nome_software} | Tópico: {hierarquia} - {titulo}\nConteúdo: {conteudo_texto}"
        stable_id_source = f"{source_path}|chunk_{chunk_ordem}|sub_{subchunk_ordem}"

        documents.append(texto_enriquecido)
        metadata: Metadata = {
            "sqlite_id": sqlite_id,
            "software": nome_software,
            "source_path": source_path,
            "chunk_ordem": chunk_ordem,
            "subchunk_ordem": subchunk_ordem,
            "subchunk_total": subchunk_total,
            "hierarquia": hierarquia,
            "titulo": titulo,
            "arquivo_hash": arquivo_hash,
            "arquivo_ingestao_em": arquivo_ingestao_em,
            "conteudo_hash": conteudo_hash,
            "conteudo_ingestao_em": conteudo_ingestao_em,
            "embedding_backend": embedding_backend,
            **embedding_model_metadata,
        }
        metadatas.append(metadata)
        ids.append(f"id_{hashlib.sha256(stable_id_source.encode('utf-8')).hexdigest()}")

    return documents, metadatas, ids


def iterar_lotes(sequence, batch_size=BATCH_SIZE):
    if batch_size <= 0:
        raise ValueError('batch_size deve ser maior que zero.')

    for index in range(0, len(sequence), batch_size):
        yield sequence[index:index + batch_size]

def popular_banco_vetorial(sqlite_path="knowledge_base.db", chroma_dir="chroma_db", device: str | None = None, batch_size: int | None = None):
    """
    Lê os chunks textuais do SQLite, gera embeddings usando SentenceTransformers
    e armazena os vetores localmente no ChromaDB.
    """
    print("=" * 60)
    print("🧠 PIPELINE DE GERAÇÃO DE EMBEDDINGS (LOCAL)")
    print("=" * 60)

    # 1. Conectar ao Banco de Dados Relacional SQLite
    if not os.path.exists(sqlite_path):
        print(f"[ERRO] O banco de dados '{sqlite_path}' não foi encontrado. Rode o pipeline de ingestão primeiro.")
        return

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Busca os chunks e os metadados do software associado
    cursor.execute("""
        SELECT t.id, s.nome_software, s.source_path, t.chunk_ordem, t.subchunk_ordem, t.subchunk_total, t.hierarquia, t.titulo, t.conteudo_texto, s.arquivo_hash, s.ingestao_em, t.conteudo_hash, t.ingestao_em
        FROM topicos t
        JOIN softwares s ON t.software_id = s.id
    """)
    rows = cursor.fetchall()

    if not rows:
        print("[AVISO] Nenhum chunk de texto encontrado no SQLite para vetorizar.")
        conn.close()
        return

    print(f"📥 Carregados {len(rows)} chunks textuais do SQLite para processamento.")

    # 2. Inicializar o Modelo de Embedding (Hugging Face)
    effective_device = _resolve_embedding_device(device)
    effective_batch_size = _resolve_batch_size(batch_size)

    resolved_model_name, embedding_provider, embedding_dimension, embedding_source = _resolve_embedding_model()
    print(f"⏳ Carregando modelo '{resolved_model_name}' na memória no device '{effective_device}' (Aguarde)...")
    model_load_start = time.perf_counter()
    model, resolved_model_name, embedding_provider, embedding_dimension, embedding_source = _load_embedding_model(effective_device)
    model.max_seq_length = 8192
    model_load_elapsed = time.perf_counter() - model_load_start
    embedding_backend = 'sentence-transformers'
    embedding_model_metadata = _build_embedding_model_metadata(
        resolved_model_name,
        embedding_provider,
        embedding_dimension,
        embedding_source,
    )
    print(f"✅ Modelo carregado com sucesso em {model_load_elapsed:.2f}s!")

    # 3. Inicializar o Banco Vetorial ChromaDB (Persistente Local)
    # Ele criará uma pasta chamada 'chroma_db' no seu projeto para salvar os índices indexados
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Cria ou obtém a coleção onde guardaremos os vetores dos manuais
    collection = chroma_client.get_or_create_collection(
        name="manuais_treinamento",
        metadata=embedding_model_metadata,
    )

    print("\n🔄 Gerando vetores e preparando pacotes...")
    documents, metadatas, ids = preparar_documentos_vetoriais(rows, embedding_backend, embedding_model_metadata)

    # 4. Gerar e persistir embeddings em lotes menores
    print(f"⚡ Calculando {len(documents)} vetores matemáticos em lotes de até {effective_batch_size} chunks no device '{effective_device}'...")
    total_lotes = (len(documents) + effective_batch_size - 1) // effective_batch_size
    vectorization_start = time.perf_counter()

    for lote_index, (document_batch, metadata_batch, id_batch) in enumerate(
        zip(
            iterar_lotes(documents, effective_batch_size),
            iterar_lotes(metadatas, effective_batch_size),
            iterar_lotes(ids, effective_batch_size),
        ),
        start=1,
    ):
        print(f"  • Processando lote {lote_index}/{total_lotes} com {len(document_batch)} chunks...")
        embeddings_raw: Any = model.encode(document_batch, show_progress_bar=False, batch_size=effective_batch_size)
        embeddings_array = _coerce_embeddings_to_array(embeddings_raw)

        print("    ↳ Gravando lote no índice do ChromaDB...")
        collection.upsert(
            embeddings=embeddings_array,
            documents=document_batch,
            metadatas=metadata_batch,
            ids=id_batch
        )

    vectorization_elapsed = time.perf_counter() - vectorization_start

    conn.close()
    print("\n" + "=" * 60)
    print("🎉 [CONCLUÍDO] Banco Vetorial criado e indexado com sucesso!")
    print(f"📁 Os vetores foram guardados na pasta local: ./{chroma_dir}")
    print(f"⏱️ Tempo total de vetorização/indexação: {vectorization_elapsed:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera embeddings e popula o ChromaDB a partir do SQLite.")
    parser.add_argument("--sqlite-path", default="knowledge_base.db", help="Caminho do banco SQLite de entrada.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Diretório persistente do ChromaDB.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device para o modelo de embeddings.")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamanho do lote usado na vetorização.")

    args = parser.parse_args()
    popular_banco_vetorial(sqlite_path=args.sqlite_path, chroma_dir=args.chroma_dir, device=args.device, batch_size=args.batch_size)
