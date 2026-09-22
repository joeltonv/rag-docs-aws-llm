from app import app


def _configure_telemetry_db(tmp_path):
    app.config["RAG_SERVICE"] = None


class _FakeRagService:
    def health(self):
        return {
            "status": "ok",
            "chunks_indexed": 2,
            "collection_name": "manuais_treinamento",
            "embedding_provider": "sentence-transformers",
            "embedding_model": "fake",
            "embedding_model_provider": "sentence-transformers",
            "embedding_model_dimension": 3,
            "region_name": "us-east-1",
            "selected_model_id": "deepseek.v3.2",
            "chat_model_id": "deepseek.v3.2",
            "temperature": 0.2,
            "top_p": 0.9,
        }

    def search(self, question, top_k=None, translate_query=None):
        return {
            "question": question,
            "top_k": top_k or 4,
            "retrieval_question": question,
            "query_translation": {"enabled": bool(translate_query), "applied": False, "original_question": question, "translated_question": question},
            "embedding_provider": "sentence-transformers",
            "collection_metadata": {"embedding_model": "fake"},
            "results": [
                {
                    "rank": 1,
                    "document": "Trecho relevante.",
                    "distance": 0.12,
                    "software": "AutoCAD",
                    "hierarquia": "1.1",
                    "titulo": "Instalação",
                    "source_path": "docs/manual.md",
                }
            ],
            "context": "Trecho relevante.",
        }

    def chat(self, question, top_k=None, model_id=None, context_override=None, sources_override=None, translate_query=None):
        del model_id, context_override, sources_override
        search_result = self.search(question, top_k=top_k, translate_query=translate_query)
        return {
            "question": question,
            "answer": "Resposta baseada no contexto.",
            "top_k": search_result["top_k"],
            "embedding_provider": search_result["embedding_provider"],
            "collection_metadata": search_result["collection_metadata"],
            "results": search_result["results"],
            "context": search_result["context"],
            "generation_metadata": {"model_id": "deepseek.v3.2", "usage": {"inputTokens": 4, "outputTokens": 8}},
            "model_id": "deepseek.v3.2",
        }


def test_root_path_is_not_exposed(tmp_path):
    _configure_telemetry_db(tmp_path)
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 404


def test_health_endpoint_returns_service_status(tmp_path):
    _configure_telemetry_db(tmp_path)
    app.config["RAG_SERVICE"] = _FakeRagService()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["chunks_indexed"] == 2
    assert response.json["selected_model_id"] == "deepseek.v3.2"


def test_search_endpoint_returns_rag_results(tmp_path):
    _configure_telemetry_db(tmp_path)
    app.config["RAG_SERVICE"] = _FakeRagService()
    client = app.test_client()

    response = client.get("/search?q=AutoCAD&top_k=2")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["question"] == "AutoCAD"
    assert response.json["top_k"] == 2
    assert response.json["results"][0]["software"] == "AutoCAD"


def test_chat_endpoint_returns_answer_payload(tmp_path):
    _configure_telemetry_db(tmp_path)
    app.config["RAG_SERVICE"] = _FakeRagService()
    client = app.test_client()

    response = client.post("/chat", json={"question": "Como instalar?", "top_k": 1, "model_id": "mistral.mistral-large-3-675b-instruct", "context": "Mesmo contexto", "results": [{"software": "AutoCAD"}]})

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["answer"] == "Resposta baseada no contexto."
    assert response.json["results"][0]["titulo"] == "Instalação"
    assert response.json["model_id"] == "deepseek.v3.2"
