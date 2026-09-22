from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bedrock_service import BedrockService, BedrockServiceError


logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9_.+-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

PORTUGUESE_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
    "se",
    "sobre",
    "um",
    "uma",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
}

GENERIC_TITLES = {
    "examples",
    "example",
    "overview",
    "notes",
    "note",
    "summary",
    "introduction",
    "references",
    "related topics",
    "see also",
    "additional information",
}


@dataclass(frozen=True)
class SectionCandidate:
    source_path: str
    titulo_documento: str
    secao: str
    titulo_secao: str
    conteudo: str
    palavras_chave: tuple[str, ...]
    categoria: str
    dificuldade: str


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_markdown(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_>#-]+", " ", cleaned)
    cleaned = cleaned.replace("|", " ")
    return _normalize_whitespace(cleaned)


def _split_sentences(text: str) -> list[str]:
    candidate = _normalize_whitespace(_strip_markdown(text))
    if not candidate:
        return []

    parts = [part.strip() for part in SENTENCE_RE.split(candidate) if part.strip()]
    return parts if parts else [candidate]


def _extract_document_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        match = HEADING_RE.match(stripped)
        if match and len(match.group(1)) == 1:
            return _normalize_whitespace(match.group("title"))

        if not stripped.startswith(("#", "-", "*", "|")) and len(stripped) <= 120:
            return _normalize_whitespace(stripped)

    return fallback


def _parse_heading(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None

    return len(match.group(1)), _normalize_whitespace(match.group("title"))


def _extract_keywords(text: str, title: str, document_title: str, max_keywords: int = 5) -> tuple[str, ...]:
    tokens = WORD_RE.findall(f"{title} {document_title} {text}")
    counter: Counter[str] = Counter()
    ordered_tokens: list[str] = []

    for token in tokens:
        normalized = token.lower().strip("._-:;()[]{}")
        if len(normalized) < 3 or normalized in PORTUGUESE_STOPWORDS:
            continue
        if normalized not in counter:
            ordered_tokens.append(normalized)
        counter[normalized] += 1

    if not ordered_tokens:
        fallback_tokens = [token for token in title.split() if token]
        return tuple(fallback_tokens[:max_keywords])

    first_positions = {token: position for position, token in enumerate(ordered_tokens)}
    ordered_tokens = sorted(ordered_tokens, key=lambda token: (-counter[token], first_positions[token]))
    return tuple(ordered_tokens[:max_keywords])


def _classify_category(title: str, text: str) -> str:
    candidate = f"{title} {text}".lower()
    if any(term in candidate for term in ("how to", "how do", "como", "create", "creating", "add", "adding", "install", "configure", "using", "edit", "editing", "modify", "modifying", "set up")):
        return "procedimento"
    if any(term in candidate for term in ("troubleshooting", "error", "diagnostic", "problem", "failure", "warning", "issue", "resolve", "fix", "bug")):
        return "diagnostico"
    if any(term in candidate for term in ("option", "settings", "preferences", "property", "properties", "parameter", "configuration", "configure")):
        return "configuracao"
    if any(term in candidate for term in ("overview", "about", "what is", "concept", "introduction", "summary", "top")):
        return "conceito"
    if any(term in candidate for term in ("example", "examples", "table", "reference", "report", "list", "library", "command")):
        return "referencia"
    return "consulta"


def _classify_difficulty(content: str, depth: int, category: str) -> str:
    score = 0
    if len(content) > 900:
        score += 1
    if len(content) > 1800:
        score += 1
    if depth >= 3:
        score += 1
    if category in {"diagnostico", "referencia"}:
        score += 1
    if "```" in content or "|" in content:
        score += 1

    if score <= 1:
        return "facil"
    if score <= 3:
        return "medio"
    return "dificil"


def _focus_phrase(candidate: SectionCandidate) -> str:
    title = candidate.titulo_secao.strip()
    if title.lower() in GENERIC_TITLES or len(title) < 4:
        if candidate.palavras_chave:
            return " ".join(candidate.palavras_chave[:2])
        return candidate.titulo_documento
    return title


def _build_question(candidate: SectionCandidate) -> str:
    focus = _focus_phrase(candidate)
    document = candidate.titulo_documento
    templates = {
        "procedimento": [
            "How does the documentation for {document} explain how to perform {focus}?",
            "What steps does the documentation describe for {focus} in {document}?",
        ],
        "diagnostico": [
            "How does the documentation for {document} guide troubleshooting for {focus}?",
            "What symptoms and corrective actions does the documentation associate with {focus} in {document}?",
        ],
        "configuracao": [
            "Which settings or options does the documentation for {document} present for {focus}?",
            "How does the documentation describe adjusting {focus} in {document}?",
        ],
        "conceito": [
            "What does the documentation for {document} explain about {focus}?",
            "What is the definition or overview of {focus} in {document}?",
        ],
        "referencia": [
            "What technical information does the documentation for {document} provide about {focus}?",
            "What reference details does the documentation collect about {focus} in {document}?",
        ],
        "consulta": [
            "What does the documentation for {document} say about {focus}?",
            "How does the documentation describe {focus} in {document}?",
        ],
    }

    variants = templates.get(candidate.categoria, templates["consulta"])
    seed = int(hashlib.sha256(f"{candidate.source_path}|{candidate.secao}".encode("utf-8")).hexdigest(), 16)
    template = variants[seed % len(variants)]
    return template.format(document=document, focus=focus)


def _build_ground_truth_answer(content: str, candidate: SectionCandidate) -> str:
    sentences = _split_sentences(content)
    if not sentences:
        return f"A documentação de {candidate.titulo_documento} trata de {candidate.titulo_secao}."

    excerpt = " ".join(sentences[:2]).strip()
    excerpt = re.sub(r"\s+", " ", excerpt)
    if len(excerpt) > 550:
        excerpt = excerpt[:547].rstrip() + "..."

    if candidate.categoria == "procedimento":
        return f"A documentação descreve o seguinte sobre {candidate.titulo_secao}: {excerpt}"

    return excerpt


def _question_generation_prompt(candidate: SectionCandidate) -> str:
    return (
        "Você está criando um dataset padrão-ouro para avaliação de RAG em documentação técnica.\n"
        "Gere exatamente uma pergunta de suporte técnico em português brasileiro, escrita como um usuário leigo ou intermediário que não conhece o título exato da seção.\n"
        "Requisitos: a pergunta não deve copiar o título da seção, não deve mencionar palavras-chave óbvias do título, deve soar natural e realista, e deve ser respondível pelo trecho fornecido.\n\n"
        f"Documento: {candidate.titulo_documento}\n"
        f"Seção: {candidate.secao}\n"
        f"Título da seção: {candidate.titulo_secao}\n"
        f"Palavras-chave sugeridas: {', '.join(candidate.palavras_chave) if candidate.palavras_chave else 'nenhuma'}\n"
        f"Categoria: {candidate.categoria}\n"
        f"Dificuldade: {candidate.dificuldade}\n\n"
        "Trecho de referência:\n"
        f"{candidate.conteudo.strip()}\n\n"
        "Retorne apenas a pergunta final em uma única linha, sem aspas, sem lista, sem numeração e sem explicações adicionais."
    )


def _iter_markdown_files(docs_root: Path) -> list[Path]:
    if docs_root.is_file():
        return [docs_root] if docs_root.suffix.lower() == ".md" else []

    markdown_files = sorted(path for path in docs_root.rglob("*.md") if path.is_file())
    return markdown_files


def _generate_question_with_llm(
    candidate: SectionCandidate,
    question_service: BedrockService,
    question_model_id: str,
    question_system_prompt: str,
) -> str:
    prompt = _question_generation_prompt(candidate)
    response = question_service.generate_text_with_metadata(
        prompt,
        system_prompt=question_system_prompt,
        model_id=question_model_id,
    )
    question = str(response.get("answer", "")).strip()
    question = question.strip('"').strip("'")
    question = _normalize_whitespace(question)
    if question.endswith(("?", ".")) and question.count("?") == 0 and question.endswith("."):
        question = question[:-1] + "?"

    if not question.endswith(("?", ".")):
        question = question + "?"

    if len(question) < 12:
        raise BedrockServiceError("Pergunta gerada pelo LLM ficou vazia ou inválida.")

    return question


def _extract_section_candidates(path: Path, repo_root: Path) -> list[SectionCandidate]:
    lines = path.read_text(encoding="utf-8").splitlines()
    document_title = _extract_document_title(lines, path.stem)
    source_path = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()

    candidates: list[SectionCandidate] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: tuple[int, str] | None = None
    current_content_lines: list[str] = []

    def finalize_current() -> None:
        nonlocal current_heading, current_content_lines
        if current_heading is None:
            return

        content = "\n".join(current_content_lines).strip()
        section_title = current_heading[1]
        if not content and current_heading[0] > 1:
            current_content_lines = []
            return

        section_path = " > ".join(title for _, title in heading_stack) or document_title
        keywords = _extract_keywords(content, section_title, document_title)
        category = _classify_category(section_title, content)
        difficulty = _classify_difficulty(content, current_heading[0], category)

        candidates.append(
            SectionCandidate(
                source_path=source_path,
                titulo_documento=document_title,
                secao=section_path,
                titulo_secao=section_title,
                conteudo=content,
                palavras_chave=keywords,
                categoria=category,
                dificuldade=difficulty,
            )
        )
        current_content_lines = []

    for line in lines:
        heading = _parse_heading(line)
        if heading is not None:
            finalize_current()
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading = heading
            current_content_lines = []
            continue

        if current_heading is not None:
            current_content_lines.append(line)

    finalize_current()

    if not candidates:
        content = "\n".join(lines).strip()
        if content:
            keywords = _extract_keywords(content, document_title, document_title)
            category = _classify_category(document_title, content)
            candidates.append(
                SectionCandidate(
                    source_path=source_path,
                    titulo_documento=document_title,
                    secao=document_title,
                    titulo_secao=document_title,
                    conteudo=content,
                    palavras_chave=keywords,
                    categoria=category,
                    dificuldade=_classify_difficulty(content, 1, category),
                )
            )

    return candidates


def _select_candidates(candidates: list[SectionCandidate], target_size: int, seed: int) -> list[SectionCandidate]:
    if target_size <= 0:
        raise ValueError("target_size deve ser maior que zero.")

    grouped: dict[str, list[SectionCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.categoria].append(candidate)

    rng = random.Random(seed)
    for group in grouped.values():
        group.sort(key=lambda item: (item.source_path, item.secao, item.titulo_secao))
        rng.shuffle(group)

    ordered_categories = sorted(grouped, key=lambda category: (-len(grouped[category]), category))
    selected: list[SectionCandidate] = []
    used_per_source: dict[str, int] = defaultdict(int)
    seen_questions: set[str] = set()

    while len(selected) < target_size:
        progress = False
        for category in ordered_categories:
            if not grouped[category]:
                continue

            choice_index = None
            for index, candidate in enumerate(grouped[category]):
                question = _build_question(candidate)
                if question in seen_questions:
                    continue
                if used_per_source[candidate.source_path] >= 2:
                    continue
                choice_index = index
                break

            if choice_index is None:
                choice_index = 0

            candidate = grouped[category].pop(choice_index)
            question = _build_question(candidate)
            if question in seen_questions:
                continue

            selected.append(candidate)
            seen_questions.add(question)
            used_per_source[candidate.source_path] += 1
            progress = True

            if len(selected) >= target_size:
                break

        if not progress:
            break

    if len(selected) < target_size:
        remaining = [candidate for group in grouped.values() for candidate in group if _build_question(candidate) not in seen_questions]
        remaining.sort(key=lambda item: (item.source_path, item.secao, item.titulo_secao))
        for candidate in remaining:
            if len(selected) >= target_size:
                break
            selected.append(candidate)

    return selected[:target_size]


def build_dataset(
    docs_root: Path,
    target_size: int = 50,
    seed: int = 42,
    repo_root: Path | None = None,
    question_service: BedrockService | None = None,
    question_model_id: str | None = None,
    question_system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Construir um dataset padrão-ouro a partir dos Markdown convertidos."""

    resolved_docs_root = docs_root.expanduser().resolve()
    resolved_repo_root = (repo_root or Path.cwd()).expanduser().resolve()

    all_candidates: list[SectionCandidate] = []
    for markdown_file in _iter_markdown_files(resolved_docs_root):
        try:
            all_candidates.extend(_extract_section_candidates(markdown_file, resolved_repo_root))
        except UnicodeDecodeError as exc:
            logger.warning("Falha ao ler %s: %s", markdown_file, exc)

    if not all_candidates:
        raise RuntimeError(f"Nenhum conteúdo Markdown elegível encontrado em '{resolved_docs_root}'.")

    selected_candidates = _select_candidates(all_candidates, target_size=target_size, seed=seed)
    dataset: list[dict[str, Any]] = []
    effective_question_service = question_service
    effective_question_model_id = question_model_id or "us.anthropic.claude-sonnet-5"
    effective_question_system_prompt = question_system_prompt or (
        "Você é um gerador de perguntas para avaliação científica. Produza perguntas variadas, realistas e semanticamente desafiadoras."
    )
    llm_questions_enabled = effective_question_service is not None

    for index, candidate in enumerate(selected_candidates, start=1):
        if llm_questions_enabled and effective_question_service is not None:
            try:
                question = _generate_question_with_llm(
                    candidate,
                    question_service=effective_question_service,
                    question_model_id=effective_question_model_id,
                    question_system_prompt=effective_question_system_prompt,
                )
            except (BedrockServiceError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                logger.warning("Falha ao gerar pergunta via LLM para %s: %s. Usando fallback determinístico.", candidate.source_path, exc)
                llm_questions_enabled = False
                effective_question_service = None
                question = _build_question(candidate)
        else:
            question = _build_question(candidate)

        answer = _build_ground_truth_answer(candidate.conteudo, candidate)
        example_id = hashlib.sha256(f"{candidate.source_path}|{candidate.secao}|{question}".encode("utf-8")).hexdigest()[:16]

        dataset.append(
            {
                "id": f"q_{example_id}",
                "pergunta": question,
                "resposta_ideal": answer,
                "source_path": candidate.source_path,
                "titulo_documento": candidate.titulo_documento,
                "secao": candidate.secao,
                "palavras_chave": list(candidate.palavras_chave),
                "categoria": candidate.categoria,
                "dificuldade": candidate.dificuldade,
                "ordem": index,
            }
        )

    return dataset


def save_dataset(dataset: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
