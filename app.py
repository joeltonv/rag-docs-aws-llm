import logging
from typing import Any

from flask import Flask, jsonify, request

from bedrock_service import BedrockServiceError
from rag_service import create_rag_service

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config.setdefault("RAG_SERVICE", None)


def _get_rag_service():
    service = app.config.get("RAG_SERVICE")
    if service is None:
        service = create_rag_service()
        app.config["RAG_SERVICE"] = service

    return service


def _extract_question() -> tuple[str, int]:
    payload = request.get_json(silent=True) or {}
    question = (
        payload.get("question")
        or payload.get("q")
        or request.form.get("question", "")
        or request.args.get("question", "")
        or request.args.get("q", "")
    ).strip()

    top_k_raw = payload.get("top_k") or request.args.get("top_k") or request.form.get("top_k")
    try:
        top_k = int(top_k_raw) if top_k_raw not in (None, "") else 0
    except (TypeError, ValueError):
        top_k = 0

    translate_raw = payload.get("translate_query") or request.args.get("translate_query") or request.form.get("translate_query")
    if translate_raw in (None, ""):
        translate_query = None
    else:
        translate_query = str(translate_raw).strip().lower() in {"1", "true", "yes", "on", "sim"}

    return question, top_k, translate_query


def _extract_chat_request() -> tuple[str, int, str | None, str | None, list[dict[str, Any]] | None]:
    payload = request.get_json(silent=True) or {}
    question, top_k, translate_query = _extract_question()

    model_id_raw = payload.get("model_id") or payload.get("model") or request.args.get("model_id") or request.args.get("model")
    model_id = str(model_id_raw).strip() if model_id_raw not in (None, "") else None

    context_raw = payload.get("context")
    context = str(context_raw) if context_raw is not None else None

    sources_raw = payload.get("results") or payload.get("sources")
    sources: list[dict[str, Any]] | None = None
    if isinstance(sources_raw, list):
        sources = [item for item in sources_raw if isinstance(item, dict)]

    return question, top_k, model_id, context, sources, translate_query


def _service_error_response(exc: Exception):
    message = str(exc)
    if "Coleção ChromaDB" in message or "does not exist" in message or "não encontrada" in message:
        return jsonify(
            {
                "status": "error",
                "message": message,
                "hint": "Recrie o índice vetorial com `python gerar_embeddings.py --sqlite-path knowledge_base.db --chroma-dir chroma_db`.",
            }
        ), 503

    return jsonify({"status": "error", "message": message}), 500


@app.get("/health")
def health():
    service = _get_rag_service()
    return jsonify(service.health())


@app.get("/search")
def search():
    question, top_k, translate_query = _extract_question()
    if not question:
        return jsonify({"status": "error", "message": "Informe a pergunta via q ou question."}), 400

    service = _get_rag_service()
    try:
        result = service.search(question, top_k=top_k or None, translate_query=translate_query)
    except (BedrockServiceError, RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.exception("Falha ao executar busca RAG.")
        return _service_error_response(exc)

    return jsonify({"status": "ok", **result})


@app.post("/chat")
def chat():
    question, top_k, model_id, context_override, sources_override, translate_query = _extract_chat_request()
    if not question:
        return jsonify({"status": "error", "message": "Informe a pergunta via JSON, form ou query string."}), 400

    service = _get_rag_service()
    try:
        result = service.chat(
            question,
            top_k=top_k or None,
            model_id=model_id,
            context_override=context_override,
            sources_override=sources_override,
            translate_query=translate_query,
        )
    except (BedrockServiceError, RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.exception("Falha ao executar chat RAG.")
        return _service_error_response(exc)

    return jsonify({"status": "ok", **result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
