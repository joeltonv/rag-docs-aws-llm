from dataclasses import dataclass

from rag_service import RagSettings, RagService


class _FakeEmbeddingBackend:
    model_name = "fake-model"
    provider = "sentence-transformers"
    dimension = 3
    source = "local"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass
class _FakeCollection:
    metadata: dict

    def query(self, **_kwargs):
        del _kwargs
        return {
            "documents": [["Trecho A sobre instalação.", "Trecho B sobre atualização."]],
            "metadatas": [[
                {"software": "AutoCAD", "hierarquia": "1.1", "titulo": "Instalação", "source_path": "docs/manual.md"},
                {"software": "AutoCAD", "hierarquia": "1.2", "titulo": "Atualização", "source_path": "docs/manual.md"},
            ]],
            "distances": [[0.1, 0.3]],
        }

    def count(self):
        return 2


class _FakeChromaClient:
    def __init__(self):
        self.collection = _FakeCollection(metadata={"embedding_model": "fake-model"})

    def get_collection(self, name):
        assert name == "manuais_treinamento"
        return self.collection


class _FakeBedrockService:
    def __init__(self):
        self.settings = type("Settings", (), {"chat_model_id": "anthropic.claude-3-haiku-20240307-v1:0"})()
        self.calls = []
        self.translation_calls = []

    def generate_text_with_metadata(self, prompt, system_prompt=None, model_id=None):
        self.translation_calls.append({"prompt": prompt, "system_prompt": system_prompt, "model_id": model_id})
        return {
            "answer": "How to install?",
            "model_id": model_id or "translation-model",
            "generation_metadata": {},
        }

    def generate_answer(self, question, context, sources=None, system_prompt=None, _model_id=None):
        del _model_id
        self.calls.append({
            "question": question,
            "context": context,
            "sources": sources,
            "system_prompt": system_prompt,
        })
        return "Resposta final."


def _settings() -> RagSettings:
    return RagSettings(
        chroma_dir="chroma_db",
        collection_name="manuais_treinamento",
        top_k=4,
        max_context_chars=1000,
        embedding_provider="sentence-transformers",
        local_embedding_model_path="modelos/bge-m3",
        local_embedding_device="cpu",
        bedrock_embedding_model_id="amazon.titan-embed-text-v2:0",
        bedrock_embedding_dimensions=1024,
        bedrock_chat_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        query_translation_enabled=True,
        query_translation_model_id="deepseek.v3.2",
        query_translation_system_prompt="Você traduz consultas técnicas para inglês.",
        system_prompt="Você é um assistente técnico.",
    )


def test_search_returns_context_and_ranked_results():
    embedding_backend = _FakeEmbeddingBackend()
    service = RagService(
        settings=_settings(),
        embedding_backend=embedding_backend,
        bedrock_service=_FakeBedrockService(),
        chroma_client=_FakeChromaClient(),
    )

    result = service.search("Como instalar?", top_k=2)

    assert result["question"] == "Como instalar?"
    assert result["retrieval_question"] == "How to install?"
    assert result["top_k"] == 2
    assert result["results"][0]["software"] == "AutoCAD"
    assert "Trecho A sobre instalação." in result["context"]
    assert embedding_backend.calls[0] == ["How to install?"]


def test_chat_uses_bedrock_service_with_built_context(caplog):
    bedrock_service = _FakeBedrockService()
    embedding_backend = _FakeEmbeddingBackend()
    service = RagService(
        settings=_settings(),
        embedding_backend=embedding_backend,
        bedrock_service=bedrock_service,
        chroma_client=_FakeChromaClient(),
    )

    caplog.set_level("INFO")
    result = service.chat("Como atualizar?")

    assert result["answer"] == "Resposta final."
    assert result["retrieval_question"] == "How to install?"
    assert bedrock_service.calls[0]["question"] == "Como atualizar?"
    assert "Trecho A sobre instalação." in bedrock_service.calls[0]["context"]
    assert "RAG chat: recuperados 2 chunks" in caplog.text
