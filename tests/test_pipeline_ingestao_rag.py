from pathlib import Path

from pipeline_ingestao_rag import _split_fallback_chunks, extract_markdown_data, initialize_database, save_to_sql_automated


def test_extract_markdown_data_supports_header_based_markdown(tmp_path: Path):
    markdown = tmp_path / "manual.md"
    markdown.write_text(
        """1. AutoCAD (Autodesk)\n\n1.1. Suporte Técnico e Erros Operacionais\n\n1.1.1. O software fechou sozinho?\n    Resposta do Bot: Reinicie o aplicativo.\n\n# Observações Gerais\nTexto livre em markdown.\n""",
        encoding="utf-8",
    )

    nome_arquivo, softwares, arquivo_hash, ingestao_em = extract_markdown_data(str(markdown))

    assert nome_arquivo == "manual.md"
    assert arquivo_hash
    assert ingestao_em.endswith("Z")
    assert softwares[0]["nome_software"] == "AutoCAD (Autodesk)"
    assert softwares[0]["arquivo_hash"] == arquivo_hash
    assert softwares[0]["topicos"]

    topico = softwares[0]["topicos"][0]
    assert topico["titulo"] == "O software fechou sozinho?"
    assert topico["conteudo_hash"]
    assert topico["ingestao_em"] == ingestao_em


def test_save_to_sql_automated_persists_audit_fields(tmp_path: Path):
    db_path = tmp_path / "knowledge.db"
    conn = initialize_database(str(db_path))

    topicos = [{
        "chunk_ordem": 1,
        "subchunk_ordem": 1,
        "subchunk_total": 1,
        "hierarquia": "1.1.1 - Suporte Técnico e Erros Operacionais",
        "titulo": "O software fechou sozinho?",
        "texto": "1.1.1. O software fechou sozinho?\nResposta do Bot: Reinicie o aplicativo.",
        "conteudo_hash": "abc123",
        "ingestao_em": "2026-07-04T12:00:00Z",
    }]

    save_to_sql_automated(
        conn,
        "AutoCAD",
        "manual.md",
        topicos,
        arquivo_hash="def456",
        ingestao_em="2026-07-04T12:00:00Z",
    )

    cursor = conn.cursor()
    cursor.execute("SELECT arquivo_hash, ingestao_em FROM softwares WHERE nome_software = ?", ("AutoCAD",))
    software_row = cursor.fetchone()
    assert software_row == ("def456", "2026-07-04T12:00:00Z")

    cursor.execute("SELECT conteudo_hash, ingestao_em FROM topicos WHERE titulo = ?", ("O software fechou sozinho?",))
    topico_row = cursor.fetchone()
    assert topico_row == ("abc123", "2026-07-04T12:00:00Z")

    conn.close()


def test_extract_markdown_data_applies_hard_cap_to_structured_chunks(tmp_path: Path):
    markdown = tmp_path / "manual_estruturado_longo.md"
    long_paragraph = (
        "Esta secção detalha o comportamento do sistema em condições normais e em falhas de execução. "
        "A explicação é propositalmente extensa para forçar a divisão em subchunks sem quebrar o contexto semântico. "
    ) * 8
    markdown.write_text(
        f"""1. AutoCAD (Autodesk)\n\n1.1. Suporte Técnico e Erros Operacionais\n\n1.1.1. O software fechou sozinho?\n{long_paragraph}\n""",
        encoding="utf-8",
    )

    _, softwares, _, _ = extract_markdown_data(str(markdown), max_chunk_chars=240, chunk_overlap_sentences=1)

    topicos = softwares[0]["topicos"]
    assert len(topicos) > 1
    assert all(len(topico["texto"]) <= 240 for topico in topicos)
    assert all(topico["chunk_ordem"] == 1 for topico in topicos)
    assert [topico["subchunk_ordem"] for topico in topicos] == list(range(1, len(topicos) + 1))
    assert all(topico["subchunk_total"] == len(topicos) for topico in topicos)
    assert topicos[0]["hierarquia"].startswith("1.1.1")
    assert "Parte 1/" in topicos[0]["hierarquia"]


def test_extract_markdown_data_uses_plain_title_as_software_header(tmp_path: Path):
    markdown = tmp_path / "manual_sem_marcador.md"
    markdown.write_text(
        """AutoCAD (Autodesk)\n\n1.1. Suporte Técnico e Erros Operacionais\n\n1.1.1. O aplicativo fechou sozinho?\n    Resposta do Bot: Reinicie o software.\n""",
        encoding="utf-8",
    )

    _, softwares, _, _ = extract_markdown_data(str(markdown))

    assert softwares[0]["nome_software"] == "AutoCAD (Autodesk)"
    assert softwares[0]["topicos"]


def test_extract_markdown_data_splits_unstructured_text_into_fallback_chunks(tmp_path: Path):
    markdown = tmp_path / "manual_sem_estrutura.md"
    long_paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 25
    markdown.write_text(
        "\n\n".join([
            "Manual Corporativo",
            long_paragraph,
            long_paragraph,
            long_paragraph,
            long_paragraph,
        ]),
        encoding="utf-8",
    )

    _, softwares, _, _ = extract_markdown_data(str(markdown))

    assert softwares[0]["nome_software"] == "Manual Corporativo"
    assert len(softwares[0]["topicos"]) >= 2
    assert all(len(topico["texto"]) <= 1200 for topico in softwares[0]["topicos"])


def test_split_fallback_chunks_uses_sentence_overlap():
    lines = [
        "Alpha beta. Gamma delta.",
        "",
        "Epsilon zeta.",
    ]

    chunks = _split_fallback_chunks(lines, max_chars=30, overlap_sentences=1)

    assert len(chunks) >= 2
    assert "Gamma delta." in chunks[0]
    assert chunks[1].startswith("Gamma delta.")


def test_split_fallback_chunks_preserves_code_blocks():
    code_lines = [
        "Antes do bloco.",
        "",
        "```bash",
        "echo 'linha 1'",
        "echo 'linha 2'",
        "echo 'linha 3'",
        "echo 'linha 4'",
        "```",
        "",
        "Depois do bloco.",
    ]

    chunks = _split_fallback_chunks(code_lines, max_chars=60, overlap_sentences=1)

    code_chunks = [chunk for chunk in chunks if "```bash" in chunk]
    assert code_chunks
    assert all(chunk.count("```") == 2 for chunk in code_chunks)


def test_split_fallback_chunks_preserves_table_blocks():
    table_lines = [
        "Antes da tabela.",
        "",
        "| Coluna | Valor |",
        "| --- | --- |",
        "| Linha 1 | A |",
        "| Linha 2 | B |",
        "| Linha 3 | C |",
        "| Linha 4 | D |",
        "",
        "Depois da tabela.",
    ]

    chunks = _split_fallback_chunks(table_lines, max_chars=70, overlap_sentences=1)

    table_chunks = [chunk for chunk in chunks if "| Coluna | Valor |" in chunk]
    assert table_chunks
    assert all("| --- | --- |" in chunk for chunk in table_chunks)
