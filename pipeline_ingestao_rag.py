import argparse
import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict


NUMERIC_HEADER_RE = re.compile(r'^(?P<number>(?:\d+\.)+)\s+(?P<title>.+)$')
MARKDOWN_HEADER_RE = re.compile(r'^(?P<level>#{1,6})\s+(?P<title>.+)$')
DEFAULT_MAX_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP_SENTENCES = 1
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
MIN_FRAGMENT_CHARS = 80


class HeaderInfo(TypedDict):
    level: int
    number: Optional[str]
    title: str


class ChunkInfo(TypedDict):
    chunk_level: int
    headers_snapshot: Dict[int, HeaderInfo]
    lines: List[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _parse_header(line: str) -> Optional[HeaderInfo]:
    stripped = line.strip()
    markdown_match = MARKDOWN_HEADER_RE.match(stripped)
    if markdown_match:
        return {
            'level': len(markdown_match.group('level')),
            'number': None,
            'title': markdown_match.group('title').strip(),
        }

    if stripped != line:
        return None

    numeric_match = NUMERIC_HEADER_RE.match(stripped)
    if numeric_match:
        return {
            'level': len([segment for segment in numeric_match.group('number').split('.') if segment]),
            'number': numeric_match.group('number').rstrip('.'),
            'title': numeric_match.group('title').strip(),
        }

    return None


def _looks_like_software_title(text: str) -> bool:
    candidate = text.strip()
    if not candidate or len(candidate) > 80:
        return False

    if candidate.endswith(('.', ':', ';', '!', '?')):
        return False

    if not any(char.isalpha() for char in candidate):
        return False

    if candidate.count(' ') > 10:
        return False

    return True


def _split_long_fragment(text: str, max_chars: int) -> List[str]:
    candidate = text.strip()
    if not candidate:
        return []

    if len(candidate) <= max_chars:
        return [candidate]

    word_parts = candidate.split()
    if len(word_parts) <= 1:
        return [candidate[:max_chars].strip()]

    chunks: List[str] = []
    current_chunk = ''

    for word in word_parts:
        next_chunk = f"{current_chunk} {word}".strip() if current_chunk else word
        if len(next_chunk) <= max_chars:
            current_chunk = next_chunk
            continue

        if current_chunk:
            chunks.append(current_chunk)
        current_chunk = word

        if len(current_chunk) > max_chars:
            chunks.extend(current_chunk[index:index + max_chars] for index in range(0, len(current_chunk), max_chars))
            current_chunk = ''

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_sentence_units(text: str, max_chars: int) -> List[str]:
    candidate = re.sub(r'\s+', ' ', text.strip())
    if not candidate:
        return []

    sentence_parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(candidate) if part.strip()]
    if len(sentence_parts) > 1:
        units: List[str] = []
        for part in sentence_parts:
            units.extend(_split_long_fragment(part, max_chars))
        return units

    if len(candidate) <= max_chars:
        return [candidate]

    return _split_long_fragment(candidate, max_chars)


def _is_fence_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('```') or stripped.startswith('~~~')


def _fence_marker(line: str) -> Optional[str]:
    stripped = line.strip()
    if stripped.startswith('```'):
        return '```'
    if stripped.startswith('~~~'):
        return '~~~'
    return None


def _is_table_separator_line(line: str) -> bool:
    parts = [part.strip() for part in line.strip().strip('|').split('|') if part.strip()]
    if not parts:
        return False

    return all(re.fullmatch(r':?-{3,}:?', part) is not None for part in parts)


def _looks_like_table_block(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False

    first_line = lines[index].strip()
    second_line = lines[index + 1].strip()
    return '|' in first_line and _is_table_separator_line(second_line)


def _split_markdown_table_block(block_lines: List[str], max_chars: int) -> List[str]:
    if len(block_lines) <= 2:
        return ['\n'.join(block_lines).strip()]

    header_line = block_lines[0].rstrip()
    separator_line = block_lines[1].rstrip()
    body_lines = [line.rstrip() for line in block_lines[2:] if line.strip()]

    chunks: List[str] = []
    current_rows: List[str] = []

    def render_table(rows: List[str]) -> str:
        return '\n'.join([header_line, separator_line, *rows]).strip()

    for row in body_lines:
        candidate_rows = current_rows + [row]
        candidate_text = render_table(candidate_rows)
        if len(candidate_text) <= max_chars:
            current_rows = candidate_rows
            continue

        if current_rows:
            chunks.append(render_table(current_rows))
            current_rows = [row]
            continue

        chunks.append(render_table([row]))
        current_rows = []

    if current_rows:
        chunks.append(render_table(current_rows))

    return chunks


def _split_fenced_code_block(block_lines: List[str], max_chars: int) -> List[str]:
    return ['\n'.join(block_lines).strip()]


def _split_markdown_aware_units(lines: List[str], max_chars: int) -> List[Tuple[str, str]]:
    units: List[Tuple[str, str]] = []
    index = 0

    while index < len(lines):
        stripped_line = lines[index].strip()
        if not stripped_line:
            index += 1
            continue

        if _is_fence_line(stripped_line):
            fence = _fence_marker(stripped_line)
            if fence is None:
                index += 1
                continue
            block_start = index
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                index += 1
            if index < len(lines):
                index += 1
            block_lines = lines[block_start:index]
            code_blocks = _split_fenced_code_block(block_lines, max_chars)
            units.extend(('code', block) for block in code_blocks if block)
            continue

        if _looks_like_table_block(lines, index):
            block_start = index
            index += 2
            while index < len(lines) and lines[index].strip() and '|' in lines[index]:
                index += 1
            block_lines = lines[block_start:index]
            table_blocks = _split_markdown_table_block(block_lines, max_chars)
            units.extend(('table', block) for block in table_blocks if block)
            continue

        block_start = index
        index += 1
        while index < len(lines) and lines[index].strip() and not _is_fence_line(lines[index]) and not _looks_like_table_block(lines, index):
            index += 1

        paragraph = ' '.join(line.strip() for line in lines[block_start:index] if line.strip())
        if paragraph:
            for sentence_unit in _split_sentence_units(paragraph, max_chars):
                units.append(('text', sentence_unit))

    return units


def _render_markdown_units(units: List[Tuple[str, str]]) -> str:
    return '\n\n'.join(text for _, text in units).strip()


def _looks_fragmentary(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.endswith(('.', '!', '?', ':', ';')):
        return False

    return len(stripped) < MIN_FRAGMENT_CHARS


def _split_fallback_chunks(lines: List[str], max_chars: int = DEFAULT_MAX_CHUNK_CHARS, overlap_sentences: int = DEFAULT_CHUNK_OVERLAP_SENTENCES) -> List[str]:
    if overlap_sentences < 0:
        raise ValueError('overlap_sentences must be greater than or equal to zero')

    markdown_units = _split_markdown_aware_units(lines, max_chars)
    if not markdown_units:
        return []

    base_chunks: List[List[Tuple[str, str]]] = []
    current_chunk_units: List[Tuple[str, str]] = []

    for unit in markdown_units:
        unit_type, unit_text = unit
        if not unit_text:
            continue

        if unit_type != 'text':
            if current_chunk_units:
                base_chunks.append(current_chunk_units)
                current_chunk_units = []

            base_chunks.append([unit])
            continue

        candidate_units = current_chunk_units + [unit]
        candidate_text = _render_markdown_units(candidate_units)

        if current_chunk_units and len(candidate_text) > max_chars:
            base_chunks.append(current_chunk_units)
            current_chunk_units = [unit]
            continue

        if len(unit_text.strip()) < MIN_FRAGMENT_CHARS and current_chunk_units:
            current_chunk_units = candidate_units
            continue

        current_chunk_units = candidate_units if candidate_text else [unit]

    if current_chunk_units:
        base_chunks.append(current_chunk_units)

    if not base_chunks:
        return []

    merged_base_chunks: List[List[Tuple[str, str]]] = []
    for chunk_units in base_chunks:
        chunk_text = _render_markdown_units(chunk_units)
        if merged_base_chunks and _looks_fragmentary(chunk_text):
            previous_chunk_text = _render_markdown_units(merged_base_chunks[-1])
            if '```' in chunk_text or '~~~' in chunk_text:
                merged_base_chunks.append(chunk_units)
                continue

            combined_text = f'{previous_chunk_text}\n\n{chunk_text}'.strip()
            if len(chunk_text) < MIN_FRAGMENT_CHARS // 2 or len(combined_text) <= max_chars:
                merged_base_chunks[-1].extend(chunk_units)
                continue

        merged_base_chunks.append(chunk_units)

    base_chunks = merged_base_chunks

    if overlap_sentences == 0 or len(base_chunks) == 1:
        return [_render_markdown_units(chunk_units) for chunk_units in base_chunks]

    chunks: List[str] = []
    for index, chunk_units in enumerate(base_chunks):
        if index == 0:
            chunks.append(_render_markdown_units(chunk_units))
            continue

        previous_chunk_units = base_chunks[index - 1]
        text_overlap = [unit for unit in previous_chunk_units if unit[0] == 'text'][-overlap_sentences:] if overlap_sentences else []
        chosen_text = ''

        for overlap_size in range(len(text_overlap), 0, -1):
            candidate_units = text_overlap[-overlap_size:] + chunk_units
            candidate_text = _render_markdown_units(candidate_units)
            if len(candidate_text) <= max_chars:
                chosen_text = candidate_text
                break

        if not chosen_text:
            chosen_text = _render_markdown_units(chunk_units)

        chunks.append(chosen_text)

    return chunks


def _resolve_max_chunk_chars(max_chunk_chars: Optional[int]) -> int:
    if max_chunk_chars is not None:
        effective_max_chunk_chars = max_chunk_chars
    else:
        max_chunk_chars_value = os.getenv('RAG_MAX_CHUNK_CHARS', str(DEFAULT_MAX_CHUNK_CHARS))
        try:
            effective_max_chunk_chars = int(max_chunk_chars_value)
        except ValueError as exc:
            raise ValueError('RAG_MAX_CHUNK_CHARS deve ser um inteiro positivo.') from exc

    if effective_max_chunk_chars <= 0:
        raise ValueError('max_chunk_chars deve ser maior que zero.')

    return effective_max_chunk_chars


def _resolve_chunk_overlap_sentences(chunk_overlap_sentences: Optional[int]) -> int:
    if chunk_overlap_sentences is not None:
        effective_overlap_sentences = chunk_overlap_sentences
    else:
        overlap_value = os.getenv('RAG_CHUNK_OVERLAP_SENTENCES', str(DEFAULT_CHUNK_OVERLAP_SENTENCES))
        try:
            effective_overlap_sentences = int(overlap_value)
        except ValueError as exc:
            raise ValueError('RAG_CHUNK_OVERLAP_SENTENCES deve ser um inteiro maior ou igual a zero.') from exc

    if effective_overlap_sentences < 0:
        raise ValueError('chunk_overlap_sentences deve ser maior ou igual a zero.')

    return effective_overlap_sentences


def _split_text_with_hard_cap(text: str, max_chars: int, overlap_sentences: int) -> List[str]:
    candidate = text.strip()
    if not candidate:
        return []

    if len(candidate) <= max_chars:
        return [candidate]

    line_chunks = _split_fallback_chunks(candidate.splitlines(), max_chars=max_chars, overlap_sentences=overlap_sentences)
    if line_chunks:
        return line_chunks

    return _split_long_fragment(candidate, max_chars)


def _format_split_hierarquia(base_hierarquia: str, subchunk_ordem: int, subchunk_total: int) -> str:
    if subchunk_total <= 1:
        return base_hierarquia

    suffix = f'Parte {subchunk_ordem}/{subchunk_total}'
    if base_hierarquia:
        return f'{base_hierarquia} | {suffix}'

    return suffix


def _header_label(header: Optional[HeaderInfo]) -> str:
    if not header:
        return ''

    number = header.get('number')
    title = header.get('title') or ''
    if number:
        return f'{number} - {title}' if title else number
    return title


def _build_chunk_metadata(
    headers_snapshot: Dict[int, HeaderInfo],
    chunk_level: int,
    file_name: str,
) -> Tuple[str, str, str]:
    software_header = headers_snapshot.get(1)
    section_header = headers_snapshot.get(2)
    chunk_header = headers_snapshot.get(chunk_level)

    if software_header is not None:
        nome_software = software_header['title']
    else:
        nome_software = os.path.splitext(os.path.basename(file_name))[0]

    if chunk_header is not None and chunk_header['title']:
        titulo = chunk_header['title']
    elif section_header is not None:
        titulo = _header_label(section_header)
    else:
        titulo = nome_software

    hierarquia: str
    if chunk_header and chunk_header.get('number'):
        if section_header and section_header.get('title'):
            chunk_number = chunk_header['number'] or ''
            section_title = section_header['title']
            hierarquia = f"{chunk_number} - {section_title}"
        else:
            hierarquia = chunk_header['number'] or ''
    else:
        hierarquia_partes = [
            _header_label(software_header),
            _header_label(section_header),
            _header_label(chunk_header),
        ]
        hierarquia = ' > '.join(parte for parte in hierarquia_partes if parte)

    return nome_software, hierarquia, titulo


def _build_topic_records(
    nome_software: str,
    hierarquia: str,
    titulo: str,
    texto: str,
    source_timestamp: str,
    chunk_ordem: int,
    max_chunk_chars: int,
    chunk_overlap_sentences: int,
) -> List[Dict[str, Any]]:
    split_texts = _split_text_with_hard_cap(texto, max_chunk_chars, chunk_overlap_sentences)
    if not split_texts:
        return []

    total_subchunks = len(split_texts)
    topic_records: List[Dict[str, Any]] = []

    for subchunk_ordem, subchunk_text in enumerate(split_texts, start=1):
        topic_records.append({
            'nome_software': nome_software,
            'hierarquia': _format_split_hierarquia(hierarquia, subchunk_ordem, total_subchunks),
            'titulo': titulo,
            'texto': subchunk_text,
            'conteudo_hash': _sha256_text(subchunk_text),
            'ingestao_em': source_timestamp,
            'chunk_ordem': chunk_ordem,
            'subchunk_ordem': subchunk_ordem,
            'subchunk_total': total_subchunks,
        })

    return topic_records


def initialize_database(db_path="knowledge_base.db"):
    """Cria a estrutura de tabelas SQL e garante chaves estrangeiras ativas."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Habilita o suporte a chaves estrangeiras
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Tabela 1: Metadados do Software
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS softwares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_software TEXT,
            nome_arquivo TEXT,
            source_path TEXT UNIQUE,
            arquivo_hash TEXT,
            ingestao_em TEXT
        )
    ''')

    # Tabela 2: Chunks de Tópicos Estruturados
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topicos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            software_id INTEGER,
            source_path TEXT,
            chunk_ordem INTEGER,
            subchunk_ordem INTEGER,
            subchunk_total INTEGER,
            hierarquia TEXT,
            titulo TEXT,
            conteudo_texto TEXT,
            conteudo_hash TEXT,
            ingestao_em TEXT,
            FOREIGN KEY (software_id) REFERENCES softwares (id) ON DELETE CASCADE
        )
    ''')

    cursor.execute("PRAGMA table_info(softwares)")
    software_columns = {row[1] for row in cursor.fetchall()}
    if 'arquivo_hash' not in software_columns:
        cursor.execute("ALTER TABLE softwares ADD COLUMN arquivo_hash TEXT")
    if 'ingestao_em' not in software_columns:
        cursor.execute("ALTER TABLE softwares ADD COLUMN ingestao_em TEXT")
    if 'source_path' not in software_columns:
        cursor.execute("ALTER TABLE softwares ADD COLUMN source_path TEXT")

    cursor.execute("PRAGMA table_info(topicos)")
    topico_columns = {row[1] for row in cursor.fetchall()}
    if 'source_path' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN source_path TEXT")
    if 'chunk_ordem' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN chunk_ordem INTEGER")
    if 'subchunk_ordem' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN subchunk_ordem INTEGER")
    if 'subchunk_total' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN subchunk_total INTEGER")
    if 'conteudo_hash' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN conteudo_hash TEXT")
    if 'ingestao_em' not in topico_columns:
        cursor.execute("ALTER TABLE topicos ADD COLUMN ingestao_em TEXT")

    conn.commit()
    return conn


def _normalize_source_path(source_path: Optional[str], fallback_name: str) -> str:
    candidate = source_path or fallback_name
    return os.path.normpath(candidate).replace('\\', '/')


def _discover_markdown_files(source_root: str) -> List[str]:
    if os.path.isfile(source_root):
        return [source_root] if source_root.lower().endswith('.md') else []

    markdown_files: List[str] = []
    for current_root, _, files in os.walk(source_root):
        for file_name in files:
            if file_name.lower().endswith('.md'):
                markdown_files.append(os.path.join(current_root, file_name))

    return sorted(markdown_files)

def extract_markdown_data(
    markdown_path,
    max_chunk_chars: Optional[int] = None,
    chunk_overlap_sentences: Optional[int] = None,
):
    """
    Analisa o arquivo Markdown com base na hierarquia de cabeçalhos e converte
    o conteúdo em chunks estruturados, preservando rastreabilidade.
    """
    with open(markdown_path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    headers = [header for line in lines if (header := _parse_header(line))]
    if headers:
        header_levels = [header['level'] for header in headers if header['level'] is not None]
        candidate_levels = [level for level in header_levels if level <= 3]
        chunk_level = max(candidate_levels) if candidate_levels else max(header_levels)
    else:
        chunk_level = 1

    software_blocks: List[Dict[str, Any]] = []
    current_software: Optional[Dict[str, Any]] = None
    current_chunk: Optional[ChunkInfo] = None
    headers_snapshot: Dict[int, HeaderInfo] = {}
    source_timestamp = _utc_now_iso()
    source_hash = _sha256_text('\n'.join(lines))
    fallback_software_name = os.path.splitext(os.path.basename(markdown_path))[0]
    effective_max_chunk_chars = _resolve_max_chunk_chars(max_chunk_chars)
    effective_chunk_overlap_sentences = _resolve_chunk_overlap_sentences(chunk_overlap_sentences)
    current_chunk_ordem = 0

    def finalize_chunk() -> None:
        nonlocal current_chunk, current_software, current_chunk_ordem
        chunk = current_chunk
        if chunk is None:
            return

        content = '\n'.join(chunk['lines']).strip()
        if not content:
            current_chunk = None
            return

        nome_software, hierarquia, titulo = _build_chunk_metadata(
            chunk['headers_snapshot'],
            chunk['chunk_level'],
            markdown_path,
        )

        if current_software is None:
            current_software = {
                'nome_software': nome_software,
                'source_path': _normalize_source_path(markdown_path, os.path.basename(markdown_path)),
                'arquivo_hash': source_hash,
                'ingestao_em': source_timestamp,
                'topicos': [],
                '_source_lines': [],
            }
        elif current_software.get('nome_software') in {'', fallback_software_name} and nome_software:
            current_software['nome_software'] = nome_software

        current_chunk_ordem += 1
        current_software['topicos'].extend(
            _build_topic_records(
                current_software['nome_software'],
                hierarquia,
                titulo,
                content,
                source_timestamp,
                current_chunk_ordem,
                effective_max_chunk_chars,
                effective_chunk_overlap_sentences,
            )
        )
        current_chunk = None

    def finalize_current_software() -> None:
        nonlocal current_software, current_chunk_ordem
        if not current_software:
            return

        if not current_software['topicos']:
            fallback_chunks = _split_fallback_chunks(
                current_software.get('_source_lines', []),
                max_chars=effective_max_chunk_chars,
                overlap_sentences=effective_chunk_overlap_sentences,
            )
            if fallback_chunks:
                for chunk_index, fallback_text in enumerate(fallback_chunks, start=1):
                    current_chunk_ordem += 1
                    current_software['topicos'].append({
                        'hierarquia': f'Chunk {chunk_index}',
                        'titulo': current_software['nome_software'],
                        'texto': fallback_text,
                        'conteudo_hash': _sha256_text(fallback_text),
                        'ingestao_em': source_timestamp,
                        'chunk_ordem': current_chunk_ordem,
                        'subchunk_ordem': 1,
                        'subchunk_total': 1,
                    })

        current_software.pop('_source_lines', None)
        software_blocks.append(current_software)
        current_software = None
        current_chunk_ordem = 0

    for raw_line in lines:
        stripped_line = raw_line.strip()
        header = _parse_header(raw_line)

        if not stripped_line:
            if current_software is not None:
                current_software.setdefault('_source_lines', []).append(raw_line)
            continue

        if header:
            header_level = header['level'] or 1
            if header_level == 1:
                finalize_chunk()
                finalize_current_software()
                current_software = {
                    'nome_software': header['title'],
                    'source_path': _normalize_source_path(markdown_path, os.path.basename(markdown_path)),
                    'arquivo_hash': source_hash,
                    'ingestao_em': source_timestamp,
                    'topicos': [],
                    '_source_lines': [raw_line],
                }
            elif current_software is None:
                current_software = {
                    'nome_software': os.path.splitext(os.path.basename(markdown_path))[0],
                    'source_path': _normalize_source_path(markdown_path, os.path.basename(markdown_path)),
                    'arquivo_hash': source_hash,
                    'ingestao_em': source_timestamp,
                    'topicos': [],
                    '_source_lines': [raw_line],
                }
            elif header_level != chunk_level:
                current_software.setdefault('_source_lines', []).append(raw_line)

            for level in sorted(list(headers_snapshot.keys()), reverse=True):
                if level >= header_level:
                    headers_snapshot.pop(level, None)
            headers_snapshot[header_level] = header

            if current_chunk is not None and header_level <= chunk_level:
                finalize_chunk()

            if header_level == chunk_level:
                current_chunk = {
                    'chunk_level': chunk_level,
                    'headers_snapshot': dict(headers_snapshot),
                    'lines': [raw_line],
                }

            continue

        if current_software is None and _looks_like_software_title(stripped_line):
            current_software = {
                'nome_software': stripped_line,
                'source_path': _normalize_source_path(markdown_path, os.path.basename(markdown_path)),
                'arquivo_hash': source_hash,
                'ingestao_em': source_timestamp,
                'topicos': [],
                '_source_lines': [],
            }
            continue

        if current_chunk is not None:
            current_chunk['lines'].append(raw_line)
        elif current_software is not None:
            current_software.setdefault('_source_lines', []).append(raw_line)

    finalize_chunk()
    finalize_current_software()

    if not software_blocks:
        fallback_text = '\n'.join(lines).strip()
        software_blocks.append({
            'nome_software': os.path.splitext(os.path.basename(markdown_path))[0],
            'source_path': _normalize_source_path(markdown_path, os.path.basename(markdown_path)),
            'arquivo_hash': source_hash,
            'ingestao_em': source_timestamp,
            'topicos': [{
                'hierarquia': 'Chunk 1',
                'titulo': 'Conteúdo do Dataset',
                'texto': fallback_text,
                'conteudo_hash': _sha256_text(fallback_text),
                'ingestao_em': source_timestamp,
                'chunk_ordem': 1,
                'subchunk_ordem': 1,
                'subchunk_total': 1,
            }],
        })

    return os.path.basename(markdown_path), software_blocks, source_hash, source_timestamp


def save_to_sql_automated(conn, nome_software, nome_arquivo, topicos_estruturados, arquivo_hash=None, ingestao_em=None, source_path=None):
    """
    Remove o software anterior pelo source_path e deixa o ON DELETE CASCADE
    do SQLite remover os tópicos associados.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    ingestao_em = ingestao_em or _utc_now_iso()
    arquivo_hash = arquivo_hash or ''
    source_key = _normalize_source_path(source_path, nome_arquivo)

    try:
        cursor.execute("SELECT id FROM softwares WHERE source_path = ?", (source_key,))
        row = cursor.fetchone()

        if row:
            software_id = row[0]
            print(f"🔄 Detectada alteração/atualização para '{source_key}'. Limpando chunks antigos...")

            cursor.execute("DELETE FROM softwares WHERE id = ?", (software_id,))

        cursor.execute(
            '''
            INSERT INTO softwares (nome_software, nome_arquivo, source_path, arquivo_hash, ingestao_em)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (nome_software, nome_arquivo, source_key, arquivo_hash, ingestao_em)
        )

        cursor.execute("SELECT id FROM softwares WHERE source_path = ?", (source_key,))
        software_id = cursor.fetchone()[0]

        for topico in topicos_estruturados:
            cursor.execute(
                '''
                INSERT INTO topicos (software_id, source_path, chunk_ordem, subchunk_ordem, subchunk_total, hierarquia, titulo, conteudo_texto, conteudo_hash, ingestao_em)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    software_id,
                    source_key,
                    topico.get("chunk_ordem"),
                    topico.get("subchunk_ordem"),
                    topico.get("subchunk_total"),
                    topico["hierarquia"],
                    topico["titulo"],
                    topico["texto"],
                    topico.get("conteudo_hash", ''),
                    topico.get("ingestao_em", ingestao_em),
                )
            )

        conn.commit()
        print(f"[SUCESSO] Base de dados atualizada para o software '{nome_software}'.\n")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"[ERRO BANCO DE DADOS] Falha ao processar {nome_software}: {e}\n")


def purge_software_records_for_file(conn, nome_arquivo):
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    source_key = _normalize_source_path(nome_arquivo, nome_arquivo)
    cursor.execute("DELETE FROM softwares WHERE source_path = ? OR nome_arquivo = ?", (source_key, nome_arquivo))
    conn.commit()

def pipeline_ingestao_rag(
    source_path,
    db_path="knowledge_base.db",
    max_chunk_chars: Optional[int] = None,
    chunk_overlap_sentences: Optional[int] = None,
):
    caminho_absoluto = os.path.abspath(source_path)

    if not os.path.exists(caminho_absoluto):
        print(f"\n[ERRO] O caminho especificado NÃO foi encontrado! ({caminho_absoluto})")
        return

    conn = initialize_database(db_path)
    arquivos = _discover_markdown_files(caminho_absoluto)

    if os.path.isfile(caminho_absoluto) and not caminho_absoluto.lower().endswith('.md'):
        print(f"[AVISO] O caminho informado não é um arquivo .md válido: {caminho_absoluto}")
        conn.close()
        return

    print("\n============================================================")
    print("🚀 INICIANDO PIPELINE DE INGESTÃO AUTOMATIZADA (RAG)")
    print("============================================================")
    print(f"Caminho alvo: {caminho_absoluto}")
    print(f"Arquivos detectados: {len(arquivos)}")
    print("-" * 60)

    if len(arquivos) == 0:
        print("[AVISO] Nenhum arquivo .md encontrado no caminho especificado.")
        conn.close()
        return

    for caminho_completo in arquivos:
        arquivo = os.path.basename(caminho_completo)
        source_key = os.path.relpath(caminho_completo, caminho_absoluto) if os.path.isdir(caminho_absoluto) else arquivo
        source_key = _normalize_source_path(source_key, arquivo)

        if os.path.getsize(caminho_completo) == 0:
            print(f"[IGNORADO] O arquivo '{arquivo}' está vazio.")
            continue

        try:
            _nome_arquivo, softwares_processados, arquivo_hash, ingestao_em = extract_markdown_data(
                caminho_completo,
                max_chunk_chars=max_chunk_chars,
                chunk_overlap_sentences=chunk_overlap_sentences,
            )
            for bloco_software in softwares_processados:
                save_to_sql_automated(
                    conn,
                    bloco_software['nome_software'],
                    source_key,
                    bloco_software['topicos'],
                    arquivo_hash=bloco_software.get('arquivo_hash', arquivo_hash),
                    ingestao_em=bloco_software.get('ingestao_em', ingestao_em),
                    source_path=bloco_software.get('source_path', source_key),
                )
        except (OSError, ValueError, sqlite3.Error) as e:
            print(f"[ERRO CRÍTICO] Falha ao processar o arquivo {arquivo}: {e}")
            continue

    conn.close()
    print("[CONCLUÍDO] O Banco de dados está sincronizado com os arquivos processados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pipeline de ingestão RAG para arquivos Markdown.'
    )
    parser.add_argument(
        'source',
        nargs='?',
        default='dataset_chatbot.md',
        help='Diretório de entrada ou arquivo .md a ser processado.'
    )
    parser.add_argument(
        '--db',
        default='knowledge_base.db',
        help='Caminho do banco SQLite de saída.'
    )
    parser.add_argument(
        '--max-chunk-chars',
        type=int,
        default=None,
        help='Limite máximo de caracteres por chunk; chunks maiores são divididos em subchunks.'
    )
    parser.add_argument(
        '--chunk-overlap-sentences',
        type=int,
        default=None,
        help='Número de sentenças reaproveitadas entre subchunks.'
    )

    args = parser.parse_args()
    pipeline_ingestao_rag(
        args.source,
        db_path=args.db,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap_sentences=args.chunk_overlap_sentences,
    )
