from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import html2text
from bs4 import BeautifulSoup, Tag
import nltk
from nltk.tokenize import PunktSentenceTokenizer


TEXTUAL_EXTENSIONS = {".htm", ".html", ".xhtml", ".xml", ".txt"}
DEFAULT_OUTPUT_ROOT = "docs_markdown"
_NOISE_SELECTOR = "nav, header, footer, aside, script, style, noscript, form, button, iframe"
_PREBLOCK_PLACEHOLDER_PREFIX = "PREBLOCK_PLACEHOLDER_"


def _ensure_punkt_resource() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


def _build_sentence_tokenizer() -> PunktSentenceTokenizer:
    _ensure_punkt_resource()
    tokenizer = PunktSentenceTokenizer()
    tokenizer_params = getattr(tokenizer, "_params")
    tokenizer_params.abbrev_types = {
        "art",
        "cap",
        "cod",
        "dr",
        "dra",
        "ed",
        "etc",
        "ex",
        "fig",
        "i.e",
        "kg",
        "km",
        "m",
        "mm",
        "mr",
        "mrs",
        "no",
        "nos",
        "p",
        "pag",
        "pp",
        "ref",
        "sr",
        "sra",
        "srta",
        "v",
    }
    return tokenizer


SENTENCE_TOKENIZER = _build_sentence_tokenizer()


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    output_path: Path
    markdown: str


def iter_textual_documents(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in TEXTUAL_EXTENSIONS:
            yield root
        return

    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in TEXTUAL_EXTENSIONS:
            yield candidate


def _normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _normalize_markdown_links(markdown: str) -> str:
    return re.sub(r"\]\(<([^>]+)>\)", r"](\1)", markdown)


def _html_int_attribute(raw_value: object, default: int = 1) -> int:
    if raw_value is None:
        return default

    if isinstance(raw_value, list):
        if not raw_value:
            return default
        raw_value = raw_value[0]

    if raw_value is None:
        return default

    if isinstance(raw_value, (int, str)):
        return int(raw_value)

    return int(str(raw_value))


def _table_to_markdown(table: Tag) -> str:
    rows: List[List[str]] = []
    active_rowspans: dict[int, dict[str, int | str]] = {}

    for row in table.find_all("tr"):
        cells: List[str] = []
        row_cells = list(row.find_all(["th", "td"]))
        cell_index = 0
        column_index = 0

        while cell_index < len(row_cells) or column_index in active_rowspans:
            if column_index in active_rowspans:
                span = active_rowspans[column_index]
                cells.append(str(span["text"]))
                remaining = int(span["remaining"]) - 1
                if remaining <= 0:
                    del active_rowspans[column_index]
                else:
                    span["remaining"] = remaining
                column_index += 1
                continue

            if cell_index >= len(row_cells):
                break

            cell = row_cells[cell_index]
            cell_index += 1
            cell_text = " ".join(cell.stripped_strings).replace("\n", " ")

            raw_colspan = cell.get("colspan")
            colspan = _html_int_attribute(raw_colspan)

            raw_rowspan = cell.get("rowspan")
            rowspan = _html_int_attribute(raw_rowspan)

            for offset in range(colspan):
                value = cell_text if offset == 0 else ""
                cells.append(value)
                if rowspan > 1:
                    active_rowspans[column_index + offset] = {"text": value, "remaining": rowspan - 1}
            column_index += colspan

        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]

    header = normalized_rows[0]
    separator = ["---"] * width
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []

    def render_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(cell.strip() or " " for cell in row) + " |"

    rendered = [render_row(header), render_row(separator)]
    rendered.extend(render_row(row) for row in body)
    return "\n".join(rendered)


def _table_looks_like_header_row(cells: Sequence[str]) -> bool:
    if len(cells) < 2:
        return False

    texts = [cell.strip() for cell in cells]
    if any(not text for text in texts):
        return False

    if any(any(char.isdigit() for char in text) for text in texts):
        return False

    header_markers = {
        "item", "item no.", "qty.", "quantity", "part number", "description",
        "symbol", "example", "value", "valor", "categoria", "category",
    }
    normalized = [text.lower() for text in texts]
    if any(text in header_markers for text in normalized):
        return True

    return len(set(normalized)) == len(normalized)


def _table_to_key_value_markdown(rows: List[List[str]]) -> str:
    rendered_rows: List[str] = []
    for row in rows:
        if not row:
            continue

        key = row[0].strip() or " "
        value = " ".join(cell.strip() for cell in row[1:] if cell.strip()) or " "
        rendered_rows.append(f"- {key}: {value}")

    return "\n".join(rendered_rows)


def _remove_noise(soup: BeautifulSoup) -> None:
    for element in soup.select(_NOISE_SELECTOR):
        element.decompose()

    for element in soup.select("[role='navigation'], [class*='breadcrumb'], [class*='related-links'], [id*='breadcrumb'], [id*='nav'], [id*='menu']"):
        element.decompose()


def _inject_table_placeholders(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    table_replacements: List[Tuple[str, str]] = []
    for index, table in enumerate(list(soup.find_all("table")), start=1):
        table_rows = [[" ".join(cell.stripped_strings).replace("\n", " ") for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        if table_rows and not _table_looks_like_header_row(table_rows[0]):
            markdown_table = _table_to_key_value_markdown(table_rows)
        else:
            markdown_table = _table_to_markdown(table)
        if not markdown_table:
            table.decompose()
            continue

        placeholder = f"TABLE_PLACEHOLDER_{index}"
        table_replacements.append((placeholder, markdown_table))
        table.replace_with(soup.new_string(f"\n\n{placeholder}\n\n"))

    return table_replacements


def _inject_preformatted_placeholders(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    preformatted_replacements: List[Tuple[str, str]] = []

    for index, tag in enumerate(list(soup.find_all("pre")), start=1):
        raw_text = tag.get_text("\n", strip=False)
        if not raw_text.strip():
            tag.decompose()
            continue

        placeholder = f"{_PREBLOCK_PLACEHOLDER_PREFIX}{index}"
        preformatted_replacements.append((placeholder, raw_text))
        tag.replace_with(soup.new_string(f"\n\n{placeholder}\n\n"))

    return preformatted_replacements


def _render_preformatted_block(raw_text: str) -> str:
    block_text = raw_text.strip("\n")
    return f"```\n{block_text}\n```"


def _html_to_markdown(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    _remove_noise(soup)
    preformatted_replacements = _inject_preformatted_placeholders(soup)
    table_replacements = _inject_table_placeholders(soup)

    if soup.title and soup.title.string:
        title_text = soup.title.string.strip()
    else:
        first_heading = soup.find(["h1", "h2"])
        title_text = first_heading.get_text(" ", strip=True) if first_heading else ""

    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.single_line_break = True
    converter.ignore_links = False
    converter.ignore_images = False
    converter.protect_links = True
    converter.wrap_links = False
    converter.skip_internal_links = False
    converter.ul_item_mark = "-"
    converter.emphasis_mark = "*"

    markdown = converter.handle(str(soup))

    for placeholder, table_markdown in table_replacements:
        markdown = markdown.replace(placeholder, table_markdown)

    for placeholder, raw_text in preformatted_replacements:
        markdown = markdown.replace(placeholder, _render_preformatted_block(raw_text))

    markdown = _normalize_markdown_links(markdown)
    markdown = _normalize_whitespace(markdown)

    if title_text and not markdown.startswith("# "):
        markdown = f"# {title_text}\n\n{markdown}" if markdown else f"# {title_text}"

    return markdown.strip() + "\n"


def _relative_document_path(source_path: Path, source_root: Path | None) -> Path:
    if source_root is not None:
        try:
            return source_path.relative_to(source_root)
        except ValueError:
            pass

    return Path(source_path.name)


def convert_documentation_file(source_path: Path, output_root: Path, source_root: Path | None = None) -> ConversionResult:
    output_path = (output_root / _relative_document_path(source_path, source_root)).with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
    markdown = _html_to_markdown(raw_text)
    output_path.write_text(markdown, encoding="utf-8")

    return ConversionResult(source_path=source_path, output_path=output_path, markdown=markdown)


def _convert_documentation_file_worker(args: Tuple[str, str, str | None]) -> ConversionResult:
    source_path_raw, output_root_raw, source_root_raw = args
    source_path = Path(source_path_raw)
    output_root = Path(output_root_raw)
    source_root = Path(source_root_raw) if source_root_raw is not None else None
    return convert_documentation_file(source_path, output_root, source_root)


def convert_documentation_tree(source_root: Path, output_root: Path, max_workers: int | None = None) -> List[ConversionResult]:
    results: List[ConversionResult] = []

    if source_root.is_file():
        if source_root.suffix.lower() in TEXTUAL_EXTENSIONS:
            results.append(convert_documentation_file(source_root, output_root, source_root.parent))
        return results

    document_paths = list(iter_textual_documents(source_root))
    if not document_paths:
        return results

    worker_count = max_workers or None
    worker_args = [(str(source_path), str(output_root), str(source_root)) for source_path in document_paths]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for result in executor.map(_convert_documentation_file_worker, worker_args):
            results.append(result)

    return results
