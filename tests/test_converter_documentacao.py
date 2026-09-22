from pathlib import Path

from converter_documentacao import convert_documentation_file, convert_documentation_tree


def test_convert_documentation_file_preserves_semantic_structure(tmp_path: Path):
    source = tmp_path / "docs" / "DraftSightSW" / "topic.htm"
    source.parent.mkdir(parents=True)
    source.write_text(
        """<!doctype html>
        <html>
          <head><title>Importar plantas baixas HomeByMe</title></head>
          <body>
            <nav>Menu lateral</nav>
            <main>
              <h1>Importar plantas baixas HomeByMe</h1>
              <p><a href="https://home.by.me/">HomeByMe</a> é um serviço online.</p>
              <img src="diagram.png" alt="Diagrama" />
              <ul><li>Janelas</li><li>Paredes</li></ul>
              <table>
                <tr><th>Item</th><th>Saída</th></tr>
                <tr><td>Janelas</td><td>Blocos</td></tr>
              </table>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = convert_documentation_file(source, tmp_path / "docs_markdown", tmp_path / "docs")

    markdown = result.output_path.read_text(encoding="utf-8")
    assert result.output_path.name == "topic.md"
    assert "# Importar plantas baixas HomeByMe" in markdown
    assert "[HomeByMe](https://home.by.me/)" in markdown
    assert "![Diagrama](diagram.png)" in markdown
    assert "- Janelas" in markdown
    assert "| Item | Saída |" in markdown
    assert "Menu lateral" not in markdown


def test_convert_documentation_tree_mirrors_nested_paths(tmp_path: Path):
    source_root = tmp_path / "docs"
    nested = source_root / "DraftSight" / "toolpalette"
    nested.mkdir(parents=True)
    (nested / "about.htm").write_text("<html><body><h1>Título</h1><p>Texto</p></body></html>", encoding="utf-8")

    results = convert_documentation_tree(source_root, tmp_path / "docs_markdown")

    assert len(results) == 1
    assert results[0].output_path == tmp_path / "docs_markdown" / "DraftSight" / "toolpalette" / "about.md"
    assert results[0].output_path.read_text(encoding="utf-8").startswith("# Título")


def test_convert_documentation_file_preserves_rowspan_tables(tmp_path: Path):
    source = tmp_path / "table.htm"
    source.write_text(
        """<html><body>
        <table>
          <tr><th>Categoria</th><th>Item</th><th>Valor</th></tr>
          <tr><td rowspan=\"2\">Especificações</td><td>Peso</td><td>10kg</td></tr>
          <tr><td>Altura</td><td>20cm</td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    result = convert_documentation_file(source, tmp_path / "out")
    markdown = result.output_path.read_text(encoding="utf-8")

    assert "| Categoria | Item | Valor |" in markdown
    assert markdown.count("Especificações") == 2
    assert "| Especificações | Peso | 10kg |" in markdown
    assert "| Especificações | Altura | 20cm |" in markdown
