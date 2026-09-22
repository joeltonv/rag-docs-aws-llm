# pyright: reportPrivateUsage=false
import sys
import types
import sqlite3
from pathlib import Path


torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


sentence_transformers_stub.SentenceTransformer = DummySentenceTransformer
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

chromadb_api_stub = types.ModuleType("chromadb.api")
sys.modules.setdefault("chromadb.api", chromadb_api_stub)

chromadb_api_types_stub = types.ModuleType("chromadb.api.types")
chromadb_api_types_stub.Metadata = dict
sys.modules.setdefault("chromadb.api.types", chromadb_api_types_stub)

import gerar_embeddings


def test_preparar_documentos_vetoriais_uses_position_based_ids():
    rows = [
        (1, "AutoCAD", "docs/manual.md", 3, 2, 5, "1.1.1 - Instalação", "Passo A", "Texto A", "hash1", "2026-07-04T12:00:00Z", "conteudo1", "2026-07-04T12:00:00Z"),
        (99, "AutoCAD", "docs/manual.md", 3, 2, 5, "1.1.1 - Instalação", "Passo A", "Texto A", "hash1", "2026-07-04T12:00:00Z", "conteudo1", "2026-07-04T12:00:00Z"),
    ]

    documents, metadatas, ids = gerar_embeddings.preparar_documentos_vetoriais(
        rows,
        "sentence-transformers",
        {
            "embedding_model": "x",
            "embedding_model_provider": "y",
            "embedding_model_dimension": 3,
            "embedding_model_source": "local",
        },
    )  # pyright: ignore[reportPrivateUsage]

    assert documents[0] == documents[1]
    assert metadatas[0]["chunk_ordem"] == 3
    assert metadatas[0]["subchunk_ordem"] == 2
    assert ids[0] == ids[1]


def test_popular_banco_vetorial_sets_max_seq_length_and_uses_position_ids(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "knowledge.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE softwares (id INTEGER PRIMARY KEY AUTOINCREMENT, nome_software TEXT, nome_arquivo TEXT, source_path TEXT UNIQUE, arquivo_hash TEXT, ingestao_em TEXT)"
    )
    cursor.execute(
        "CREATE TABLE topicos (id INTEGER PRIMARY KEY AUTOINCREMENT, software_id INTEGER, source_path TEXT, chunk_ordem INTEGER, subchunk_ordem INTEGER, subchunk_total INTEGER, hierarquia TEXT, titulo TEXT, conteudo_texto TEXT, conteudo_hash TEXT, ingestao_em TEXT)"
    )
    cursor.execute(
        "INSERT INTO softwares (nome_software, nome_arquivo, source_path, arquivo_hash, ingestao_em) VALUES (?, ?, ?, ?, ?)",
        ("AutoCAD", "manual.md", "docs/manual.md", "hash1", "2026-07-04T12:00:00Z"),
    )
    software_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO topicos (software_id, source_path, chunk_ordem, subchunk_ordem, subchunk_total, hierarquia, titulo, conteudo_texto, conteudo_hash, ingestao_em) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (software_id, "docs/manual.md", 3, 2, 5, "1.1.1 - Instalação", "Passo A", "Texto A", "conteudo1", "2026-07-04T12:00:00Z"),
    )
    conn.commit()
    conn.close()

    class FakeModel:
        def __init__(self):
            self.max_seq_length = None
            self.calls = []

        def encode(self, document_batch, show_progress_bar=False, batch_size=None):
            self.calls.append((list(document_batch), show_progress_bar, batch_size))
            return [[0.1, 0.2, 0.3] for _ in document_batch]

    class FakeCollection:
        def __init__(self):
            self.upserts = []
            self.kwargs = {}

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

    class FakeClient:
        def __init__(self):
            self.collection = FakeCollection()

        def get_or_create_collection(self, **kwargs):
            self.collection.kwargs.clear()
            self.collection.kwargs.update(kwargs)
            return self.collection

    fake_model = FakeModel()
    fake_client = FakeClient()

    monkeypatch.setattr(
        "gerar_embeddings._load_embedding_model",
        lambda device="cpu": (fake_model, str(tmp_path / "modelos" / "bge-m3"), "sentence-transformers", 1024, "local"),
    )
    monkeypatch.setattr("gerar_embeddings.chromadb.PersistentClient", lambda path: fake_client)

    (tmp_path / "modelos" / "bge-m3").mkdir(parents=True)

    gerar_embeddings.popular_banco_vetorial(
        sqlite_path=str(db_path),
        chroma_dir=str(tmp_path / "chroma"),
        device="cpu",
        batch_size=1,
    )

    assert fake_model.max_seq_length == 8192
    assert fake_client.collection.upserts
    assert fake_client.collection.upserts[0]["ids"][0].startswith("id_")
