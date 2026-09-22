from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.dataset_generation import build_dataset
from experiments.evaluation import ExperimentEvaluator
from experiments.metrics_reporting import ExperimentMetricsPipeline


class _FakeRagService:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int | None, bool | None]] = []
        self.chat_calls: list[dict[str, object]] = []

    def search(self, question: str, top_k: int | None = None, translate_query: bool | None = None):
        self.search_calls.append((question, top_k, translate_query))
        return {
            "question": question,
            "retrieval_question": question,
            "query_translation": {"enabled": bool(translate_query), "applied": False, "original_question": question, "translated_question": question},
            "top_k": top_k,
            "context": "Contexto recuperado.",
            "results": [
                {
                    "rank": 1,
                    "document": "Trecho principal.",
                    "distance": 0.2,
                    "metadata": {"source_path": "C:/Users/joelton/Documents/Pós-graduação NLP/12. Trabalho de Conclusão de Curso/GitHub/TCC_Pos_LLMs/docs_markdown/manual.md", "titulo": "Instalacao"},
                    "source_path": "C:/Users/joelton/Documents/Pós-graduação NLP/12. Trabalho de Conclusão de Curso/GitHub/TCC_Pos_LLMs/docs_markdown/manual.md",
                    "titulo": "Instalacao",
                    "hierarquia": "1.1",
                }
            ],
        }

    def chat(self, question: str, top_k: int | None = None, model_id: str | None = None, context_override: str | None = None, sources_override=None, translate_query: bool | None = None):
        self.chat_calls.append(
            {
                "question": question,
                "top_k": top_k,
                "model_id": model_id,
                "context_override": context_override,
                "sources_override": sources_override,
                "translate_query": translate_query,
            }
        )
        return {
            "answer": f"Resposta para {model_id}",
            "context": context_override or "",
            "retrieval_question": question,
            "query_translation": {"enabled": bool(translate_query), "applied": False, "original_question": question, "translated_question": question},
            "context_truncated": False,
            "generation_metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}},
        }


class _FakeJudgeService:
    def generate_tool_use_with_metadata(
        self,
        prompt: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, object],
        system_prompt: str | None = None,
        model_id: str | None = None,
    ):
        del prompt, tool_name, tool_description, input_schema, system_prompt
        return {
            "payload": {
                "groundedness": {"score": 5, "comment": "Alinhada ao contexto."},
                "corretude": {"score": 5, "comment": "Correta."},
                "completude": {"score": 4, "comment": "Quase completa."},
                "clareza": {"score": 5, "comment": "Clara."},
                "precisao_tecnica": {"score": 5, "comment": "Tecnica."},
                "alucinacao": {"score": 5, "comment": "Sem alucinacao."},
            },
            "answer": json.dumps(
                {
                    "groundedness": {"score": 5, "comment": "Alinhada ao contexto."},
                    "corretude": {"score": 5, "comment": "Correta."},
                    "completude": {"score": 4, "comment": "Quase completa."},
                    "clareza": {"score": 5, "comment": "Clara."},
                    "precisao_tecnica": {"score": 5, "comment": "Tecnica."},
                    "alucinacao": {"score": 5, "comment": "Sem alucinacao."},
                },
                ensure_ascii=False,
            ),
            "model_id": model_id or "judge-model",
        }


class _FakeQuestionService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text_with_metadata(self, prompt: str, system_prompt: str | None = None, model_id: str | None = None):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "model_id": model_id})
        return {
            "answer": "Como faço para ajustar as opções dessa funcionalidade?",
            "model_id": model_id or "question-model",
        }


class _AlwaysFailingQuestionService:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text_with_metadata(self, prompt: str, system_prompt: str | None = None, model_id: str | None = None):
        del prompt, system_prompt, model_id
        self.calls += 1
        raise RuntimeError("servico indisponivel")


def test_build_dataset_generates_records_from_markdown(tmp_path: Path):
    repo_root = tmp_path
    docs_root = repo_root / "docs_markdown"
    docs_root.mkdir(parents=True)
    markdown = docs_root / "manual.md"
    markdown.write_text(
        """# Manual de Teste

## Instalacao
A instalacao deve ser feita em duas etapas. Primeiro configure o ambiente. Depois confirme a validacao.

## Configuracao
As opcoes permitem ajustar o comportamento do sistema.
""",
        encoding="utf-8",
    )

    dataset = build_dataset(docs_root, target_size=2, seed=7, repo_root=repo_root)

    assert len(dataset) == 2
    assert dataset[0]["source_path"] == "docs_markdown/manual.md"
    assert dataset[0]["pergunta"]
    assert dataset[0]["resposta_ideal"]
    assert dataset[0]["palavras_chave"]


def test_build_dataset_uses_llm_question_generator_when_provided(tmp_path: Path):
    repo_root = tmp_path
    docs_root = repo_root / "docs_markdown"
    docs_root.mkdir(parents=True)
    markdown = docs_root / "manual.md"
    markdown.write_text(
        """# Manual de Teste

## Instalacao
A instalacao deve ser feita em duas etapas. Primeiro configure o ambiente. Depois confirme a validacao.
""",
        encoding="utf-8",
    )

    question_service = _FakeQuestionService()
    dataset = build_dataset(
        docs_root,
        target_size=1,
        seed=7,
        repo_root=repo_root,
        question_service=question_service,
        question_model_id="question-model",
        question_system_prompt="Gere uma pergunta neutra.",
    )

    assert len(question_service.calls) == 1
    assert dataset[0]["pergunta"] == "Como faço para ajustar as opções dessa funcionalidade?"


def test_build_dataset_disables_llm_after_first_failure(tmp_path: Path):
    repo_root = tmp_path
    docs_root = repo_root / "docs_markdown"
    docs_root.mkdir(parents=True)
    markdown = docs_root / "manual.md"
    markdown.write_text(
        """# Manual de Teste

## Instalacao
A instalacao deve ser feita em duas etapas. Primeiro configure o ambiente. Depois confirme a validacao.

## Configuracao
As opcoes permitem ajustar o comportamento do sistema.
""",
        encoding="utf-8",
    )

    question_service = _AlwaysFailingQuestionService()
    dataset = build_dataset(
        docs_root,
        target_size=2,
        seed=7,
        repo_root=repo_root,
        question_service=question_service,
        question_model_id="question-model",
        question_system_prompt="Gere uma pergunta neutra.",
    )

    assert question_service.calls == 1
    assert len(dataset) == 2
    assert all(item["pergunta"] for item in dataset)


def test_experiment_evaluator_reuses_one_retrieval_per_question(tmp_path: Path):
    fake_rag = _FakeRagService()
    evaluator = ExperimentEvaluator(rag_service=fake_rag, model_ids=("model-a", "model-b"), top_k=3)
    dataset = [
        {
            "id": "q_1",
            "pergunta": "Como instalar?",
            "resposta_ideal": "Execute duas etapas.",
            "source_path": "docs_markdown/manual.md",
            "titulo_documento": "Manual de Teste",
            "secao": "Manual de Teste > Instalacao",
            "categoria": "procedimento",
            "dificuldade": "facil",
        }
    ]

    output = evaluator.run(dataset=dataset, output_csv=tmp_path / "results.csv")

    assert len(fake_rag.search_calls) == 1
    assert len(fake_rag.chat_calls) == 2
    assert output.raw_results_csv.exists()
    assert output.records[0]["source_paths_chunks"]
    assert output.records[0]["tokens"] == 15
    assert output.records[0]["retrieved_rank"] == 1
    assert output.records[0]["retrieval_hit"] is True
    assert output.records[0]["hit_rate_at_k"] == 1.0
    assert output.records[0]["variacao_retrieval"] == "production"


def test_metrics_pipeline_generates_consolidated_report(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    results_path = tmp_path / "results.csv"
    consolidated_path = tmp_path / "consolidated.csv"
    report_path = tmp_path / "report.md"
    figures_dir = tmp_path / "figures"

    dataset = [
        {
            "id": "q_1",
            "pergunta": "Como instalar?",
            "resposta_ideal": "Execute duas etapas.",
            "source_path": "docs_markdown/manual.md",
            "titulo_documento": "Manual de Teste",
            "secao": "Manual de Teste > Instalacao",
            "categoria": "procedimento",
            "dificuldade": "facil",
        }
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "pergunta",
                "refraseamento",
                "variacao_retrieval",
                "variacao_descricao",
                "query_translation",
                "resposta_ideal",
                "source_path_ground_truth",
                "titulo_documento",
                "secao",
                "categoria",
                "dificuldade",
                "modelo",
                "resposta",
                "tempo_resposta_s",
                "input_tokens",
                "output_tokens",
                "tokens",
                "top_k",
                "translate_query",
                "chunks_recuperados",
                "chunks_recuperados_json",
                "source_paths_chunks",
                "similaridades_chunks",
                "similaridade_media_chunks",
                "retrieved_rank",
                "retrieval_hit",
                "hit_rate_at_k",
                "recall_at_k",
                "precision_at_k",
                "mrr",
                "context_truncated",
                "contexto",
                "generation_metadata",
                "answer_length",
                "question_length",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "q_1",
                "pergunta": "Como instalar?",
                "refraseamento": "How to install?",
                "variacao_retrieval": "production",
                "variacao_descricao": "Configuração única baseada no Top-K informado.",
                "query_translation": json.dumps({"enabled": True, "applied": True, "original_question": "Como instalar?", "translated_question": "How to install?"}, ensure_ascii=False),
                "resposta_ideal": "Execute duas etapas.",
                "source_path_ground_truth": "docs_markdown/manual.md",
                "titulo_documento": "Manual de Teste",
                "secao": "Manual de Teste > Instalacao",
                "categoria": "procedimento",
                "dificuldade": "facil",
                "modelo": "model-a",
                "resposta": "Resposta para model-a",
                "tempo_resposta_s": 0.25,
                "input_tokens": 10,
                "output_tokens": 5,
                "tokens": 15,
                "top_k": 3,
                "translate_query": True,
                "chunks_recuperados": 1,
                "chunks_recuperados_json": json.dumps(
                    [
                        {
                            "rank": 1,
                            "source_path": str((tmp_path / "docs_markdown" / "manual.md").resolve()),
                            "titulo": "Instalacao",
                            "hierarquia": "1.1",
                            "distance": 0.2,
                            "similarity": 0.833333,
                            "document": "Trecho principal.",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "source_paths_chunks": json.dumps([str((tmp_path / "docs_markdown" / "manual.md").resolve())], ensure_ascii=False),
                "similaridades_chunks": json.dumps([0.833333], ensure_ascii=False),
                "similaridade_media_chunks": 0.833333,
                "retrieved_rank": 1,
                "retrieval_hit": True,
                "hit_rate_at_k": 1.0,
                "recall_at_k": 1.0,
                "precision_at_k": 0.333333,
                "mrr": 1.0,
                "context_truncated": False,
                "contexto": "Contexto recuperado.",
                "generation_metadata": json.dumps({"usage": {"inputTokens": 10, "outputTokens": 5}}, ensure_ascii=False),
                "answer_length": 20,
                "question_length": 14,
            }
        )

    pipeline = ExperimentMetricsPipeline(
        judge_service=_FakeJudgeService(),
        judge_model_id="judge-model",
        judge_system_prompt="Avalie estritamente.",
    )

    output = pipeline.run(
        dataset_path=dataset_path,
        results_csv=results_path,
        consolidated_csv=consolidated_path,
        report_markdown=report_path,
        figures_dir=figures_dir,
    )

    assert consolidated_path.exists()
    assert report_path.exists()
    assert output["report_markdown"].name == "relatorio_comparativo.md"
    assert output["analysis_markdown"].name == "relatorio_retrieval.md"
    assert output["discussion_markdown"].name == "relatorio_generation.md"
    assert (report_path.parent / "resumo_metricas.csv").exists()
    assert (report_path.parent / "resumo_metricas_retrieval.csv").exists()
    assert (report_path.parent / "resumo_metricas_generation.csv").exists()
    assert (report_path.parent / "comparativo_variantes.csv").exists()
    assert any(path.suffix == ".png" for path in figures_dir.iterdir())
    assert any(path.suffix == ".svg" for path in figures_dir.iterdir())
    assert output["ranked_models"][0][0] == "model-a"
    assert output["records"][0]["hit_rate_at_k"] == 1.0
    assert output["records"][0]["retrieved_rank"] == 1
    assert output["records"][0]["retrieval_hit"] is True
    assert output["records"][0]["groundedness_score"] == 5
