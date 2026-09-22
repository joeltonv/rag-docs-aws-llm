from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


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
class BedrockSettings:
    region_name: str
    chat_model_id: str
    embedding_model_id: str
    embedding_dimensions: int
    temperature: float
    top_p: float
    max_tokens: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: float

    @classmethod
    def from_env(cls) -> "BedrockSettings":
        return cls(
            region_name=_env_str("AWS_REGION", _env_str("AWS_DEFAULT_REGION", "us-east-1")),
            chat_model_id=_env_str("RAG_BEDROCK_CHAT_MODEL_ID", "deepseek.v3.2"),
            embedding_model_id=_env_str("RAG_BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            embedding_dimensions=_env_int("RAG_BEDROCK_EMBEDDING_DIMENSIONS", 1024),
            temperature=_env_float("RAG_BEDROCK_TEMPERATURE", 0.2),
            top_p=_env_float("RAG_BEDROCK_TOP_P", 0.9),
            max_tokens=_env_int("RAG_BEDROCK_MAX_TOKENS", 800),
            connect_timeout_seconds=_env_float("RAG_BEDROCK_CONNECT_TIMEOUT_SECONDS", 10.0),
            read_timeout_seconds=_env_float("RAG_BEDROCK_READ_TIMEOUT_SECONDS", 90.0),
            max_attempts=_env_int("RAG_BEDROCK_MAX_ATTEMPTS", 3),
            retry_backoff_seconds=_env_float("RAG_BEDROCK_RETRY_BACKOFF_SECONDS", 0.75),
        )


class BedrockServiceError(RuntimeError):
    pass


class BedrockService:
    def __init__(self, settings: Optional[BedrockSettings] = None, client: Any | None = None) -> None:
        self.settings = settings or BedrockSettings.from_env()
        self._client = client
        self._retryable_exception_types: tuple[type[BaseException], ...] = ()

    def _load_boto3_dependencies(self) -> tuple[Any, Any, tuple[type[BaseException], ...]]:
        boto3_spec = importlib.util.find_spec("boto3")
        config_spec = importlib.util.find_spec("botocore.config")
        exceptions_spec = importlib.util.find_spec("botocore.exceptions")
        if boto3_spec is None or config_spec is None or exceptions_spec is None:
            raise BedrockServiceError(
                "boto3/botocore são necessários para usar AWS Bedrock. Instale as dependências do projeto."
            )

        boto3 = importlib.import_module("boto3")
        config_module = importlib.import_module("botocore.config")
        exceptions_module = importlib.import_module("botocore.exceptions")
        Config = config_module.Config
        BotoCoreError = exceptions_module.BotoCoreError
        ClientError = exceptions_module.ClientError
        EndpointConnectionError = exceptions_module.EndpointConnectionError

        return boto3, Config, (BotoCoreError, ClientError, EndpointConnectionError)

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client

        boto3, config_cls, retryable_exception_types = self._load_boto3_dependencies()
        self._retryable_exception_types = retryable_exception_types
        session = boto3.Session(region_name=self.settings.region_name)
        client_config = config_cls(
            connect_timeout=self.settings.connect_timeout_seconds,
            read_timeout=self.settings.read_timeout_seconds,
            retries={"max_attempts": self.settings.max_attempts, "mode": "standard"},
        )
        self._client = session.client("bedrock-runtime", config=client_config)
        return self._client

    def _read_body(self, response: Any) -> dict[str, Any]:
        body = response.get("body")
        if body is None:
            raise BedrockServiceError("Resposta do Bedrock sem corpo útil.")

        payload = body.read()
        if isinstance(payload, bytes):
            payload_text = payload.decode("utf-8")
        else:
            payload_text = str(payload)

        try:
            return json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise BedrockServiceError("Falha ao decodificar a resposta JSON do Bedrock.") from exc

    def _describe_client_error(self, exc: BaseException) -> str:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return str(exc)

        error = response.get("Error", {})
        if not isinstance(error, dict):
            return str(exc)

        code = str(error.get("Code", "")).strip()
        message = str(error.get("Message", "")).strip()
        request_metadata = response.get("ResponseMetadata", {})
        request_id = ""
        if isinstance(request_metadata, dict):
            request_id = str(request_metadata.get("RequestId", "")).strip()

        parts: list[str] = []
        if code:
            parts.append(code)
        if message:
            parts.append(message)
        if request_id:
            parts.append(f"request_id={request_id}")

        return " - ".join(parts) if parts else str(exc)

    def _is_retryable_error(self, exc: BaseException, retryable_errors: tuple[type[BaseException], ...]) -> bool:
        if isinstance(exc, retryable_errors):
            return True

        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            error = response.get("Error", {})
            if isinstance(error, dict):
                return error.get("Code") in {
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ServiceUnavailableException",
                    "ModelTimeoutException",
                    "InternalServerException",
                }

        return False

    def _log_bedrock_error(self, action: str, exc: BaseException) -> None:
        logger.warning("Erro ao chamar Bedrock %s (%s): %s", action, type(exc).__name__, str(exc))

    def _invoke_model(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client_or_create()
        retryable_errors = self._retryable_exception_types

        last_error: BaseException | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                response = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload).encode("utf-8"),
                    contentType="application/json",
                    accept="application/json",
                )
                return self._read_body(response)
            except retryable_errors as exc:
                last_error = exc
                self._log_bedrock_error("invoke_model", exc)
                if attempt >= self.settings.max_attempts or not self._is_retryable_error(exc, retryable_errors):
                    break

                sleep_seconds = min(self.settings.retry_backoff_seconds * (2 ** (attempt - 1)), 10.0)
                sleep_seconds += random.uniform(0.0, self.settings.retry_backoff_seconds)
                logger.warning(
                    "Erro transitório ao chamar Bedrock (tentativa %s/%s). Repetindo em %.2fs.",
                    attempt,
                    self.settings.max_attempts,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        error_details = self._describe_client_error(last_error) if last_error is not None else "erro desconhecido"
        raise BedrockServiceError(f"Falha ao invocar o modelo Bedrock '{model_id}': {error_details}") from last_error

    def _is_llama4_model(self, model_id: str) -> bool:
        return model_id.startswith("meta.llama4-")

    def _should_omit_temperature(self, model_id: str) -> bool:
        normalized_model_id = model_id.removeprefix("us.")
        return normalized_model_id.startswith("anthropic.claude-sonnet-5")

    def _build_converse_inference_config(self, model_id: str) -> dict[str, Any]:
        inference_config: dict[str, Any] = {"maxTokens": self.settings.max_tokens}

        if not self._should_omit_temperature(model_id):
            inference_config["temperature"] = self.settings.temperature
            inference_config["topP"] = self.settings.top_p

        return inference_config

    def _build_llama_prompt(
        self,
        question: str,
        context: str,
        sources: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        generation_prompt = self._build_generation_prompt(question, context, sources)
        system_text = (system_prompt or "Você é um assistente técnico preciso, objetivo e restrito ao conteúdo recuperado.").strip()

        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{generation_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _llama4_profile_model_id(self, model_id: str) -> str:
        return model_id if model_id.startswith("us.") else f"us.{model_id}"

    def _should_try_llama4_profile(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "accessdeniedexception",
                "validationexception",
                "resourcenotfoundexception",
                "modelnotreadyexception",
            )
        )

    def _sanitize_llama4_generation(self, text: str) -> str:
        cleaned_text = text.strip()
        for stop_token in ("<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"):
            stop_position = cleaned_text.find(stop_token)
            if stop_position != -1:
                cleaned_text = cleaned_text[:stop_position].rstrip()

        return cleaned_text

    def _invoke_llama4_native(
        self,
        model_id: str,
        question: str,
        context: str,
        sources: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_llama_prompt(question, context, sources, system_prompt=system_prompt)
        payload = {
            "prompt": prompt,
            "max_gen_len": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }

        attempted_model_ids = [model_id]
        profile_model_id = self._llama4_profile_model_id(model_id)
        if profile_model_id != model_id:
            attempted_model_ids.append(profile_model_id)

        response: dict[str, Any] | None = None
        effective_model_id = model_id
        last_error: BaseException | None = None

        for candidate_model_id in attempted_model_ids:
            try:
                response = self._invoke_model(candidate_model_id, payload)
                effective_model_id = candidate_model_id
                break
            except BedrockServiceError as exc:
                last_error = exc
                if candidate_model_id == model_id and candidate_model_id != profile_model_id and self._should_try_llama4_profile(exc):
                    logger.info(
                        "Fallback para profile Bedrock cross-region ao invocar Llama 4: %s -> %s.",
                        candidate_model_id,
                        profile_model_id,
                    )
                    continue
                raise

        if response is None:
            raise BedrockServiceError(f"Falha ao invocar o modelo Bedrock '{model_id}': {last_error}") from last_error

        generation = response.get("generation")
        if not isinstance(generation, str) or not generation.strip():
            raise BedrockServiceError("O modelo de geração do Bedrock não retornou texto utilizável.")

        answer = self._sanitize_llama4_generation(generation)
        if not answer:
            raise BedrockServiceError("O modelo de geração do Bedrock retornou apenas tokens especiais.")

        return {
            "answer": answer,
            "model_id": effective_model_id,
            "generation_metadata": self._extract_generation_metadata(response, effective_model_id),
            "raw_response": response,
        }

    def _invoke_converse(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._client_or_create()
        retryable_errors = self._retryable_exception_types

        conversation_payload: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": self._build_converse_inference_config(model_id),
        }
        if system_prompt:
            conversation_payload["system"] = [{"text": system_prompt}]
        if tool_config is not None:
            conversation_payload["toolConfig"] = tool_config

        last_error: BaseException | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                return client.converse(**conversation_payload)
            except retryable_errors as exc:
                last_error = exc
                self._log_bedrock_error("Converse", exc)
                if attempt >= self.settings.max_attempts or not self._is_retryable_error(exc, retryable_errors):
                    break

                sleep_seconds = min(self.settings.retry_backoff_seconds * (2 ** (attempt - 1)), 10.0)
                sleep_seconds += random.uniform(0.0, self.settings.retry_backoff_seconds)
                logger.warning(
                    "Erro transitório ao chamar Bedrock Converse (tentativa %s/%s). Repetindo em %.2fs.",
                    attempt,
                    self.settings.max_attempts,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise BedrockServiceError(f"Falha ao invocar o modelo Bedrock '{model_id}' via Converse.") from last_error

    def _build_tool_config(self, tool_name: str, tool_description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": tool_description,
                        "inputSchema": {"json": input_schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        }

    def _extract_tool_use_input(self, response: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []

        for part in content:
            if not isinstance(part, dict):
                continue

            tool_use = part.get("toolUse")
            if not isinstance(tool_use, dict):
                continue

            if str(tool_use.get("name", "")).strip() != tool_name:
                continue

            tool_input = tool_use.get("input")
            if isinstance(tool_input, dict):
                return tool_input

            if isinstance(tool_input, str):
                try:
                    parsed_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    return None

                if isinstance(parsed_input, dict):
                    return parsed_input

        return None

    def _extract_generation_metadata(self, response: dict[str, Any], model_id: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {"model_id": model_id}

        usage = response.get("usage")
        if isinstance(usage, dict):
            metadata["usage"] = usage

        metrics = response.get("metrics")
        if isinstance(metrics, dict):
            metadata["metrics"] = metrics

        for field in ("prompt_token_count", "generation_token_count", "stop_reason"):
            value = response.get(field)
            if value is not None:
                metadata[field] = value

        return metadata

    def embed_text(self, text: str, model_id: str | None = None) -> list[float]:
        effective_model_id = model_id or self.settings.embedding_model_id
        payload = {
            "inputText": text,
            "dimensions": self.settings.embedding_dimensions,
            "normalize": True,
        }
        response = self._invoke_model(effective_model_id, payload)

        embedding = response.get("embedding") or response.get("embeddings")
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            return [float(value) for value in embedding[0]]
        if isinstance(embedding, list):
            return [float(value) for value in embedding]

        raise BedrockServiceError("O modelo de embeddings do Bedrock não retornou um vetor válido.")

    def _build_generation_prompt(self, question: str, context: str, sources: list[dict[str, Any]] | None = None) -> str:
        source_lines: list[str] = []
        for source in sources or []:
            source_lines.append(
                f"- {source.get('software', '')} | {source.get('hierarquia', '')} | {source.get('titulo', '')}"
            )

        sources_block = "\n".join(line for line in source_lines if line).strip() or "Nenhuma fonte recuperada."

        return (
            "Você é um assistente técnico especializado em documentação corporativa. "
            "Responda apenas com base no contexto recuperado. Se a resposta não estiver presente, diga isso explicitamente.\n\n"
            f"Pergunta: {question}\n\n"
            f"Contexto:\n{context.strip() or 'Sem contexto disponível.'}\n\n"
            f"Fontes recuperadas:\n{sources_block}\n\n"
            "Resposta em português brasileiro:"
        )

    def generate_answer(
        self,
        question: str,
        context: str,
        sources: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> str:
        return self.generate_answer_with_metadata(
            question,
            context,
            sources=sources,
            system_prompt=system_prompt,
            model_id=model_id,
        )["answer"]

    def generate_answer_with_metadata(
        self,
        question: str,
        context: str,
        sources: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        effective_model_id = model_id or self.settings.chat_model_id

        if self._is_llama4_model(effective_model_id):
            return self._invoke_llama4_native(
                effective_model_id,
                question,
                context,
                sources=sources,
                system_prompt=system_prompt,
            )

        prompt = self._build_generation_prompt(question, context, sources)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        response = self._invoke_converse(effective_model_id, messages, system_prompt=system_prompt)

        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        combined_text = "".join(texts).strip()
        if not combined_text:
            raise BedrockServiceError("O modelo de geração do Bedrock não retornou texto utilizável.")

        return {
            "answer": combined_text,
            "model_id": effective_model_id,
            "generation_metadata": self._extract_generation_metadata(response, effective_model_id),
            "raw_response": response,
        }

    def _build_text_generation_prompt(self, prompt: str) -> str:
        return prompt.strip()

    def _invoke_llama4_native_text(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        prompt_text = self._build_text_generation_prompt(prompt)
        system_text = (system_prompt or "Você é um assistente preciso e objetivo.").strip()

        native_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        payload = {
            "prompt": native_prompt,
            "max_gen_len": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }

        attempted_model_ids = [model_id]
        profile_model_id = self._llama4_profile_model_id(model_id)
        if profile_model_id != model_id:
            attempted_model_ids.append(profile_model_id)

        response: dict[str, Any] | None = None
        effective_model_id = model_id
        last_error: BaseException | None = None

        for candidate_model_id in attempted_model_ids:
            try:
                response = self._invoke_model(candidate_model_id, payload)
                effective_model_id = candidate_model_id
                break
            except BedrockServiceError as exc:
                last_error = exc
                if candidate_model_id == model_id and candidate_model_id != profile_model_id and self._should_try_llama4_profile(exc):
                    logger.info(
                        "Fallback para profile Bedrock cross-region ao invocar Llama 4: %s -> %s.",
                        candidate_model_id,
                        profile_model_id,
                    )
                    continue
                raise

        if response is None:
            raise BedrockServiceError(f"Falha ao invocar o modelo Bedrock '{model_id}': {last_error}") from last_error

        generation = response.get("generation")
        if not isinstance(generation, str) or not generation.strip():
            raise BedrockServiceError("O modelo de geração do Bedrock não retornou texto utilizável.")

        answer = self._sanitize_llama4_generation(generation)
        if not answer:
            raise BedrockServiceError("O modelo de geração do Bedrock retornou apenas tokens especiais.")

        return {
            "answer": answer,
            "model_id": effective_model_id,
            "generation_metadata": self._extract_generation_metadata(response, effective_model_id),
            "raw_response": response,
        }

    def generate_text_with_metadata(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        effective_model_id = model_id or self.settings.chat_model_id

        if self._is_llama4_model(effective_model_id):
            return self._invoke_llama4_native_text(effective_model_id, prompt, system_prompt=system_prompt)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        response = self._invoke_converse(effective_model_id, messages, system_prompt=system_prompt)

        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        combined_text = "".join(texts).strip()
        if not combined_text:
            raise BedrockServiceError("O modelo de geração do Bedrock não retornou texto utilizável.")

        return {
            "answer": combined_text,
            "model_id": effective_model_id,
            "generation_metadata": self._extract_generation_metadata(response, effective_model_id),
            "raw_response": response,
        }

    def generate_tool_use_with_metadata(
        self,
        prompt: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        effective_model_id = model_id or self.settings.chat_model_id

        if self._is_llama4_model(effective_model_id):
            raise BedrockServiceError(
                f"Tool Use via Converse nao e suportado para o modelo '{effective_model_id}'. Use um modelo compatível com Converse."
            )

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        tool_config = self._build_tool_config(tool_name, tool_description, input_schema)
        response = self._invoke_converse(effective_model_id, messages, system_prompt=system_prompt, tool_config=tool_config)

        payload = self._extract_tool_use_input(response, tool_name)
        if payload is None:
            output = response.get("output", {})
            message = output.get("message", {}) if isinstance(output, dict) else {}
            content = message.get("content", []) if isinstance(message, dict) else []
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            combined_text = "".join(texts).strip()
            if not combined_text:
                raise BedrockServiceError("O modelo do Bedrock não retornou um toolUse utilizável nem texto recuperável.")

            try:
                parsed_text = json.loads(combined_text)
            except json.JSONDecodeError as exc:
                raise BedrockServiceError("O modelo do Bedrock não retornou um payload JSON válido para o juiz.") from exc

            if not isinstance(parsed_text, dict):
                raise BedrockServiceError("O payload do juiz precisa ser um objeto JSON.")

            payload = parsed_text

        return {
            "payload": payload,
            "answer": json.dumps(payload, ensure_ascii=False),
            "model_id": effective_model_id,
            "generation_metadata": self._extract_generation_metadata(response, effective_model_id),
            "raw_response": response,
            "tool_name": tool_name,
        }

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_id: str | None = None,
    ) -> str:
        return self.generate_text_with_metadata(prompt, system_prompt=system_prompt, model_id=model_id)["answer"]
