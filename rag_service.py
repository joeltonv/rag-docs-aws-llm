from __future__ import annotations

import json
import logging
import os
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Protocol, Sequence

from bedrock_service import BedrockService, BedrockSettings
import chromadb
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = "modelos/bge-m3"
DEFAULT_COLLECTION_NAME = "manuais_treinamento"


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


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um número.") from exc


@dataclass(frozen=True)
class RagSettings:
    chroma_dir: str
    collection_name: str
    top_k: int
    max_context_chars: int
    embedding_provider: str
    local_embedding_model_path: str
    local_embedding_device: str
    bedrock_embedding_model_id: str
    bedrock_embedding_dimensions: int
    bedrock_chat_model_id: str
    query_translation_enabled: bool
    query_translation_model_id: str
    query_translation_system_prompt: str
    system_prompt: str

    @classmethod
    def from_env(cls) -> "RagSettings":
        return cls(
            chroma_dir=_env_str("RAG_CHROMA_DIR", "chroma_db"),
            collection_name=_env_str("RAG_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
            top_k=_env_int("RAG_TOP_K", 10),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 6000),
            embedding_provider=_env_str("RAG_EMBEDDING_PROVIDER", "sentence-transformers").lower(),
            local_embedding_model_path=_env_str("RAG_LOCAL_EMBEDDING_MODEL_PATH", DEFAULT_LOCAL_EMBEDDING_MODEL),
            local_embedding_device=_env_str("RAG_LOCAL_EMBEDDING_DEVICE", "cpu").lower(),
            bedrock_embedding_model_id=_env_str("RAG_BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            bedrock_embedding_dimensions=_env_int("RAG_BEDROCK_EMBEDDING_DIMENSIONS", 1024),
            bedrock_chat_model_id=_env_str("RAG_BEDROCK_CHAT_MODEL_ID", "deepseek.v3.2"),
            query_translation_enabled=_env_str("RAG_QUERY_TRANSLATION_ENABLED", "true").lower() in {"1", "true", "yes", "on", "sim"},
            query_translation_model_id=_env_str("RAG_QUERY_TRANSLATION_MODEL_ID", "deepseek.v3.2"),
            query_translation_system_prompt=_env_str(
                "RAG_QUERY_TRANSLATION_SYSTEM_PROMPT",
                "Você traduz consultas técnicas do português brasileiro para inglês para fins de recuperação vetorial. Preserve nomes próprios, siglas, comandos, menus e termos do SolidWorks/DraftSight. Expanda a terminologia técnica quando isso ajudar a busca. Retorne somente o texto traduzido, sem aspas, sem comentários e sem explicações.",
            ),
            system_prompt=_env_str(
                "RAG_SYSTEM_PROMPT",
                "Você é um assistente técnico preciso, objetivo e restrito ao conteúdo recuperado.",
            ),
        )


class EmbeddingBackend(Protocol):
    model_name: str
    provider: str
    dimension: int
    source: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


class LocalSentenceTransformerEmbeddingBackend:
    def __init__(self, model_path: str, device: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers não está disponível neste ambiente.")

        repo_root = os.path.dirname(os.path.abspath(__file__))
        resolved_model_path = model_path
        if not os.path.isabs(resolved_model_path):
            resolved_model_path = os.path.join(repo_root, resolved_model_path)

        if not os.path.isdir(resolved_model_path):
            raise FileNotFoundError(f"Modelo local de embeddings não encontrado em '{resolved_model_path}'.")

        self.model = SentenceTransformer(resolved_model_path, device=device)
        self.model.max_seq_length = 8192
        self.model_name = resolved_model_path
        self.provider = "sentence-transformers"
        self.dimension = 1024
        self.source = "local"

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        embeddings = self.model.encode(list(texts), show_progress_bar=False)
        if hasattr(embeddings, "tolist"):
            raw_embeddings = embeddings.tolist()
        else:
            raw_embeddings = embeddings

        return [[float(value) for value in embedding] for embedding in raw_embeddings]


class BedrockEmbeddingBackend:
    def __init__(self, service: BedrockService, model_id: str, dimension: int) -> None:
        self.service = service
        self.model_name = model_id
        self.provider = "bedrock"
        self.dimension = dimension
        self.source = "aws-bedrock"

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self.service.embed_text(text, model_id=self.model_name) for text in texts]


def _resolve_embedding_backend(settings: RagSettings) -> EmbeddingBackend:
    if settings.embedding_provider in {"sentence-transformers", "local"}:
        return LocalSentenceTransformerEmbeddingBackend(settings.local_embedding_model_path, settings.local_embedding_device)

    if settings.embedding_provider == "bedrock":
        service = BedrockService(
            BedrockSettings(
                region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
                chat_model_id=settings.bedrock_chat_model_id,
                embedding_model_id=settings.bedrock_embedding_model_id,
                embedding_dimensions=settings.bedrock_embedding_dimensions,
                temperature=_env_float("RAG_BEDROCK_TEMPERATURE", 0.2),
                top_p=_env_float("RAG_BEDROCK_TOP_P", 0.9),
                max_tokens=_env_int("RAG_BEDROCK_MAX_TOKENS", 800),
                connect_timeout_seconds=_env_float("RAG_BEDROCK_CONNECT_TIMEOUT_SECONDS", 10.0),
                read_timeout_seconds=_env_float("RAG_BEDROCK_READ_TIMEOUT_SECONDS", 90.0),
                max_attempts=_env_int("RAG_BEDROCK_MAX_ATTEMPTS", 3),
                retry_backoff_seconds=_env_float("RAG_BEDROCK_RETRY_BACKOFF_SECONDS", 0.75),
            )
        )
        return BedrockEmbeddingBackend(service, settings.bedrock_embedding_model_id, settings.bedrock_embedding_dimensions)

    raise ValueError("RAG_EMBEDDING_PROVIDER deve ser 'sentence-transformers' ou 'bedrock'.")


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    document: str
    distance: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "document": self.document,
            "distance": self.distance,
            "metadata": self.metadata,
            "software": self.metadata.get("software"),
            "hierarquia": self.metadata.get("hierarquia"),
            "titulo": self.metadata.get("titulo"),
            "source_path": self.metadata.get("source_path"),
        }


class RagService:
    def __init__(
        self,
        settings: Optional[RagSettings] = None,
        embedding_backend: EmbeddingBackend | None = None,
        bedrock_service: BedrockService | None = None,
        chroma_client: Any | None = None,
    ) -> None:
        self.settings = settings or RagSettings.from_env()
        self.embedding_backend = embedding_backend or _resolve_embedding_backend(self.settings)
        self.bedrock_service = bedrock_service or BedrockService()
        self._chroma_client = chroma_client

    def _client_or_create(self) -> Any:
        if self._chroma_client is not None:
            return self._chroma_client

        if chromadb is None:
            raise RuntimeError("chromadb não está disponível neste ambiente.")

        self._chroma_client = chromadb.PersistentClient(path=self.settings.chroma_dir)
        return self._chroma_client

    def _collection(self) -> Any:
        client = self._client_or_create()
        try:
            return client.get_collection(name=self.settings.collection_name)
        except (AttributeError, KeyError, LookupError, ValueError) as exc:
            raise RuntimeError(
                f"Coleção ChromaDB '{self.settings.collection_name}' não encontrada em '{self.settings.chroma_dir}'."
            ) from exc

    def _collection_metadata(self) -> dict[str, Any]:
        collection = self._collection()
        metadata = getattr(collection, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _embed_question(self, question: str) -> list[float]:
        embedding = self.embedding_backend.encode([question])[0]
        return [float(value) for value in embedding]

    def _normalize_translation_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                translated_query = parsed.get("translated_query") or parsed.get("translation") or parsed.get("query")
                if isinstance(translated_query, str) and translated_query.strip():
                    return translated_query.strip()

        for quote in ('"', "'"):
            if cleaned.startswith(quote) and cleaned.endswith(quote) and len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()

        return cleaned

    def _translate_question_for_retrieval(self, question: str, translate_query: bool | None = None) -> dict[str, Any]:
        should_translate = self.settings.query_translation_enabled if translate_query is None else translate_query
        if not should_translate:
            return {
                "retrieval_question": question,
                "query_translation": {
                    "enabled": False,
                    "applied": False,
                    "model_id": None,
                    "original_question": question,
                    "translated_question": question,
                },
            }

        prompt = (
            "Traduza a consulta abaixo do português brasileiro para o inglês técnico usado em documentação do SolidWorks e do DraftSight. "
            "Preserve siglas, nomes de menus, comandos, recursos e termos já consagrados em inglês. "
            "Expanda a terminologia técnica quando isso melhorar a busca. Retorne apenas o texto traduzido, sem explicações.\n\n"
            f"Consulta: {question}"
        )

        try:
            response = self.bedrock_service.generate_text_with_metadata(
                prompt,
                system_prompt=self.settings.query_translation_system_prompt,
                model_id=self.settings.query_translation_model_id,
            )
            translated_question = self._normalize_translation_text(str(response.get("answer", "")))
            if not translated_question:
                translated_question = question

            return {
                "retrieval_question": translated_question,
                "query_translation": {
                    "enabled": True,
                    "applied": translated_question != question,
                    "model_id": response.get("model_id", self.settings.query_translation_model_id),
                    "original_question": question,
                    "translated_question": translated_question,
                    "generation_metadata": response.get("generation_metadata", {}),
                },
            }
        except Exception as exc:
            logger.warning("Falha ao traduzir pergunta para recuperacao; usando consulta original. %s", exc)
            return {
                "retrieval_question": question,
                "query_translation": {
                    "enabled": True,
                    "applied": False,
                    "model_id": self.settings.query_translation_model_id,
                    "original_question": question,
                    "translated_question": question,
                    "error": str(exc),
                },
            }

    def _query_collection(self, question: str, top_k: int) -> list[RetrievedChunk]:
        collection = self._collection()
        question_embedding = self._embed_question(question)
        raw_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        distances = (raw_results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievedChunk(
                    rank=index + 1,
                    document=str(document),
                    distance=float(distance) if distance is not None else None,
                    metadata=metadata,
                )
            )

        return chunks

    def _build_context(self, chunks: Sequence[RetrievedChunk]) -> tuple[str, bool]:
        context_blocks: list[str] = []
        total_chars = 0
        truncated = False

        for chunk in chunks:
            block = (
                f"[Trecho {chunk.rank}]\n"
                f"Software: {chunk.metadata.get('software', 'N/A')}\n"
                f"Hierarquia: {chunk.metadata.get('hierarquia', 'N/A')}\n"
                f"Título: {chunk.metadata.get('titulo', 'N/A')}\n"
                f"Fonte: {chunk.metadata.get('source_path', 'N/A')}\n"
                f"Distância: {chunk.distance if chunk.distance is not None else 'N/A'}\n"
                f"Conteúdo:\n{chunk.document.strip()}"
            ).strip()

            if total_chars and total_chars + len(block) > self.settings.max_context_chars:
                truncated = True
                context_blocks.append(
                    "[AVISO: Outros documentos relevantes foram omitidos por limite de contexto]"
                )
                break

            context_blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(context_blocks).strip(), truncated

    def search(self, question: str, top_k: int | None = None, translate_query: bool | None = None) -> dict[str, Any]:
        effective_top_k = top_k or self.settings.top_k
        translation_result = self._translate_question_for_retrieval(question, translate_query=translate_query)
        retrieval_question = str(translation_result.get("retrieval_question", question))
        chunks = self._query_collection(retrieval_question, effective_top_k)
        context, context_truncated = self._build_context(chunks)

        return {
            "question": question,
            "retrieval_question": retrieval_question,
            "query_translation": translation_result.get("query_translation", {}),
            "top_k": effective_top_k,
            "embedding_provider": self.embedding_backend.provider,
            "collection_metadata": self._collection_metadata(),
            "results": [chunk.to_dict() for chunk in chunks],
            "context": context,
            "context_truncated": context_truncated,
        }

    def chat(
        self,
        question: str,
        top_k: int | None = None,
        model_id: str | None = None,
        context_override: str | None = None,
        sources_override: list[dict[str, Any]] | None = None,
        translate_query: bool | None = None,
    ) -> dict[str, Any]:
        if context_override is not None or sources_override is not None:
            search_result = {
                "question": question,
                "retrieval_question": question,
                "query_translation": {
                    "enabled": False if translate_query is False else self.settings.query_translation_enabled,
                    "applied": False,
                    "model_id": None,
                    "original_question": question,
                    "translated_question": question,
                },
                "top_k": top_k or self.settings.top_k,
                "embedding_provider": self.embedding_backend.provider,
                "collection_metadata": self._collection_metadata(),
                "results": sources_override or [],
                "context": context_override or "",
                "context_truncated": False,
            }
        else:
            search_result = self.search(question, top_k=top_k, translate_query=translate_query)

        logger.info(
            "RAG chat: recuperados %s chunks, contexto_final=%s caracteres.",
            len(search_result["results"]),
            len(search_result["context"]),
        )
        if hasattr(self.bedrock_service, "generate_answer_with_metadata"):
            generation = self.bedrock_service.generate_answer_with_metadata(  # type: ignore[attr-defined]
                question=question,
                context=search_result["context"],
                sources=search_result["results"],
                system_prompt=self.settings.system_prompt,
                model_id=model_id,
            )
        else:
            generate_answer_kwargs = {
                "question": question,
                "context": search_result["context"],
                "sources": search_result["results"],
                "system_prompt": self.settings.system_prompt,
            }
            signature = inspect.signature(self.bedrock_service.generate_answer)
            if "model_id" in signature.parameters:
                generate_answer_kwargs["model_id"] = model_id
            elif "_model_id" in signature.parameters:
                generate_answer_kwargs["_model_id"] = model_id

            answer = self.bedrock_service.generate_answer(**generate_answer_kwargs)
            generation = {
                "answer": answer,
                "model_id": model_id or self.bedrock_service.settings.chat_model_id,
                "generation_metadata": {},
                "raw_response": {},
            }

        return {
            "question": question,
            "answer": generation["answer"],
            "retrieval_question": search_result.get("retrieval_question", question),
            "query_translation": search_result.get("query_translation", {}),
            "top_k": search_result["top_k"],
            "embedding_provider": search_result["embedding_provider"],
            "collection_metadata": search_result["collection_metadata"],
            "results": search_result["results"],
            "context": search_result["context"],
            "context_truncated": search_result.get("context_truncated", False),
            "generation_metadata": generation["generation_metadata"],
            "model_id": generation["model_id"],
        }

    def health(self) -> dict[str, Any]:
        try:
            collection = self._collection()
            count = collection.count()
            return {
                "status": "ok",
                "chroma_dir": self.settings.chroma_dir,
                "collection_name": self.settings.collection_name,
                "chunks_indexed": int(count),
                "embedding_provider": self.embedding_backend.provider,
                "embedding_model": self.embedding_backend.model_name,
                "embedding_model_provider": self.embedding_backend.provider,
                "embedding_model_dimension": self.embedding_backend.dimension,
                "region_name": self.bedrock_service.settings.region_name,
                "selected_model_id": self.bedrock_service.settings.chat_model_id,
                "chat_model_id": self.bedrock_service.settings.chat_model_id,
                "temperature": self.bedrock_service.settings.temperature,
                "top_p": self.bedrock_service.settings.top_p,
            }
        except (AttributeError, KeyError, LookupError, RuntimeError, ValueError) as exc:
            return {
                "status": "degraded",
                "error": str(exc),
                "chroma_dir": self.settings.chroma_dir,
                "collection_name": self.settings.collection_name,
            }


@lru_cache(maxsize=1)
def create_rag_service() -> RagService:
    return RagService()
