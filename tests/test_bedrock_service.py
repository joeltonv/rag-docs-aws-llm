import json

from bedrock_service import BedrockService, BedrockSettings


class _FakeBody:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class _FakeClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return {"body": _FakeBody(payload)}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return payload


class _FakeRetryableClientError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"RequestId": "req-123"},
        }


def test_embed_text_uses_bedrock_payload_and_parses_response():
    client = _FakeClient([{"embedding": [0.1, 0.2, 0.3]}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    embedding = service.embed_text("Pergunta de teste")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    request_payload = json.loads(client.calls[0]["body"].decode("utf-8"))
    assert request_payload["inputText"] == "Pergunta de teste"
    assert request_payload["dimensions"] == 1024


def test_generate_answer_uses_anthropic_message_payload():
    client = _FakeClient([{"output": {"message": {"content": [{"text": "Resposta final."}]}}, "usage": {"inputTokens": 10, "outputTokens": 5}, "metrics": {"latencyMs": 123}}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    answer = service.generate_answer("Como atualizar?", "Use o menu Ajuda.", sources=[{"software": "AutoCAD"}])

    assert answer == "Resposta final."
    request_payload = client.calls[0]
    assert request_payload["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert request_payload["messages"][0]["role"] == "user"
    assert "Como atualizar?" in request_payload["messages"][0]["content"][0]["text"]
    assert request_payload["inferenceConfig"]["maxTokens"] == 256


def test_generate_answer_with_metadata_returns_tokens_and_metrics():
    client = _FakeClient([{"output": {"message": {"content": [{"text": "Resposta detalhada."}]}}, "usage": {"inputTokens": 10, "outputTokens": 7}, "metrics": {"latencyMs": 456}}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    result = service.generate_answer_with_metadata("Como atualizar?", "Use o menu Ajuda.")

    assert result["answer"] == "Resposta detalhada."
    assert result["model_id"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["generation_metadata"]["usage"]["inputTokens"] == 10
    assert result["generation_metadata"]["metrics"]["latencyMs"] == 456


def test_generate_answer_omits_temperature_for_claude_sonnet_5_converse_payload():
    client = _FakeClient([{"output": {"message": {"content": [{"text": "Resposta final."}]}}, "usage": {"inputTokens": 8, "outputTokens": 4}}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="us.anthropic.claude-sonnet-5",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    result = service.generate_answer_with_metadata("Como atualizar?", "Use o menu Ajuda.")

    assert result["answer"] == "Resposta final."
    request_payload = client.calls[0]
    assert request_payload["modelId"] == "us.anthropic.claude-sonnet-5"
    assert "temperature" not in request_payload["inferenceConfig"]
    assert "topP" not in request_payload["inferenceConfig"]
    assert request_payload["inferenceConfig"]["maxTokens"] == 256


def test_generate_answer_keeps_temperature_and_top_p_for_other_converse_models():
    for model_id in (
        "deepseek.v3.2",
        "mistral.mistral-large-3-675b-instruct",
    ):
        client = _FakeClient([{"output": {"message": {"content": [{"text": "Resposta final."}]}}, "usage": {"inputTokens": 8, "outputTokens": 4}}])
        service = BedrockService(
            settings=BedrockSettings(
                region_name="us-east-1",
                chat_model_id=model_id,
                embedding_model_id="amazon.titan-embed-text-v2:0",
                embedding_dimensions=1024,
                temperature=0.2,
                top_p=0.9,
                max_tokens=256,
                connect_timeout_seconds=5.0,
                read_timeout_seconds=20.0,
                max_attempts=1,
                retry_backoff_seconds=0.1,
            ),
            client=client,
        )

        result = service.generate_answer_with_metadata("Como atualizar?", "Use o menu Ajuda.")

        assert result["answer"] == "Resposta final."
        request_payload = client.calls[0]
        assert request_payload["modelId"] == model_id
        assert request_payload["inferenceConfig"]["maxTokens"] == 256
        assert request_payload["inferenceConfig"]["temperature"] == 0.2
        assert request_payload["inferenceConfig"]["topP"] == 0.9


def test_generate_tool_use_with_metadata_uses_converse_tool_config_and_parses_payload():
    client = _FakeClient(
        [
            {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "name": "avaliar_resposta_rag",
                                    "input": {
                                        "groundedness": {"score": 5, "comment": "Ok."},
                                        "corretude": {"score": 5, "comment": "Ok."},
                                        "completude": {"score": 4, "comment": "Ok."},
                                        "clareza": {"score": 5, "comment": "Ok."},
                                        "precisao_tecnica": {"score": 5, "comment": "Ok."},
                                        "alucinacao": {"score": 5, "comment": "Ok."},
                                    },
                                }
                            }
                        ]
                    }
                },
                "usage": {"inputTokens": 11, "outputTokens": 9},
            }
        ]
    )
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    result = service.generate_tool_use_with_metadata(
        "Avalie a resposta.",
        tool_name="avaliar_resposta_rag",
        tool_description="Preenche uma avaliação estruturada de respostas de RAG com notas de 1 a 5.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        system_prompt="Seja objetivo.",
    )

    assert result["payload"]["groundedness"]["score"] == 5
    assert result["model_id"] == "anthropic.claude-3-haiku-20240307-v1:0"
    request_payload = client.calls[0]
    assert request_payload["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert request_payload["toolConfig"]["toolChoice"]["tool"]["name"] == "avaliar_resposta_rag"
    assert request_payload["toolConfig"]["tools"][0]["toolSpec"]["name"] == "avaliar_resposta_rag"
    assert result["generation_metadata"]["usage"]["inputTokens"] == 11


def test_generate_answer_uses_llama4_native_prompt_format():
    client = _FakeClient([{"generation": "Resposta final.", "prompt_token_count": 12, "generation_token_count": 5, "stop_reason": "stop"}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="meta.llama4-maverick-17b-instruct-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    result = service.generate_answer_with_metadata("Como atualizar?", "Use o menu Ajuda.", sources=[{"software": "AutoCAD"}])

    assert result["answer"] == "Resposta final."
    assert result["model_id"] == "meta.llama4-maverick-17b-instruct-v1:0"
    request_payload = client.calls[0]
    assert request_payload["modelId"] == "meta.llama4-maverick-17b-instruct-v1:0"
    native_request = json.loads(request_payload["body"].decode("utf-8"))
    assert native_request["prompt"].startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
    assert native_request["max_gen_len"] == 256
    assert result["generation_metadata"]["prompt_token_count"] == 12
    assert result["generation_metadata"]["generation_token_count"] == 5
    assert result["generation_metadata"]["stop_reason"] == "stop"


def test_generate_answer_falls_back_to_llama4_cross_region_profile():
    class _FlakyLlamaClient:
        def __init__(self):
            self.calls = []
            self._failed_once = False

        def invoke_model(self, **kwargs):
            self.calls.append(kwargs)
            if not self._failed_once:
                self._failed_once = True
                raise _FakeRetryableClientError("AccessDeniedException", "modelo base indisponível")

            return {"body": _FakeBody({"generation": "Resposta via profile."})}

        def converse(self, **kwargs):
            raise AssertionError("Converse não deve ser chamado para Llama 4")

    client = _FlakyLlamaClient()
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="meta.llama4-maverick-17b-instruct-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )
    service._retryable_exception_types = (_FakeRetryableClientError,)

    result = service.generate_answer_with_metadata("Como atualizar?", "Use o menu Ajuda.")

    assert result["answer"] == "Resposta via profile."
    assert result["model_id"] == "us.meta.llama4-maverick-17b-instruct-v1:0"
    assert client.calls[0]["modelId"] == "meta.llama4-maverick-17b-instruct-v1:0"
    assert client.calls[1]["modelId"] == "us.meta.llama4-maverick-17b-instruct-v1:0"


def test_generate_answer_strips_llama4_stop_tokens_from_response():
    client = _FakeClient([{"generation": "Resposta final.<|eot_id|>"}])
    service = BedrockService(
        settings=BedrockSettings(
            region_name="us-east-1",
            chat_model_id="meta.llama4-maverick-17b-instruct-v1:0",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=20.0,
            max_attempts=1,
            retry_backoff_seconds=0.1,
        ),
        client=client,
    )

    result = service.generate_answer("Como atualizar?", "Use o menu Ajuda.")

    assert result == "Resposta final."
