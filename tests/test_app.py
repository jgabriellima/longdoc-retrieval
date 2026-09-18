from pathlib import Path

import pytest

from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.app.ingest import (
    document_id_from_path,
    ingest_file,
    ingest_if_needed,
    read_text_file,
)
from longdoc_retrieval.app.store import open_service, resolve_db_path, resolve_document_id


def test_resolve_db_path_cli_wins_over_env_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    env = {"LONGDOC_DB": str(tmp_path / "from-env.db")}
    cli = str(tmp_path / "from-cli.db")
    assert resolve_db_path(cli, env=env, cwd=tmp_path) == (tmp_path / "from-cli.db").resolve()
    assert resolve_db_path(None, env=env, cwd=tmp_path) == (tmp_path / "from-env.db").resolve()
    assert resolve_db_path(None, env={}, cwd=tmp_path) == (tmp_path / ".longdoc" / "index.db").resolve()


def test_open_service_creates_parent_directory(tmp_path: Path):
    db = tmp_path / ".longdoc" / "index.db"
    service = open_service(db)
    assert db.exists()
    assert service.list_documents() == []


def test_resolve_document_id_implicit_when_unique(ingested):
    assert resolve_document_id(ingested, None) == "convenio"


def test_resolve_document_id_errors_when_empty(service):
    with pytest.raises(UserError, match="no documents in index"):
        resolve_document_id(service, None)


def test_resolve_document_id_errors_when_ambiguous(service):
    from longdoc_retrieval.domain.document import Document

    service.ingest(Document(document_id="a", content="# A\n\nOne.\n"))
    service.ingest(Document(document_id="b", content="# B\n\nTwo.\n"))
    with pytest.raises(UserError, match="2 documents in index"):
        resolve_document_id(service, None)


def test_resolve_document_id_explicit_missing(service):
    with pytest.raises(UserError, match="document not found: ghost"):
        resolve_document_id(service, "ghost")


def test_document_id_from_path_uses_stem():
    assert document_id_from_path(Path("/tmp/contrato.txt"), None) == "contrato"
    assert document_id_from_path(Path("/tmp/contrato.txt"), "custom") == "custom"


def test_read_text_file_rejects_pdf(tmp_path: Path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(UserError, match="PDF"):
        read_text_file(pdf)


def test_read_text_file_rejects_binary(tmp_path: Path):
    blob = tmp_path / "dump.bin"
    blob.write_bytes(b"hello\x00world")
    with pytest.raises(UserError, match="binary"):
        read_text_file(blob)


def test_ingest_file_replaces_and_ingest_if_needed_skips(service, sample_file: Path):
    first = ingest_file(service, sample_file, "convenio")
    assert first.replaced is False
    second = ingest_file(service, sample_file, "convenio")
    assert second.replaced is True
    skipped = ingest_if_needed(service, sample_file, "convenio")
    assert skipped.ingested is False
    other = sample_file.with_name("outro.md")
    other.write_text("# Outro\n\nTexto.\n", encoding="utf-8")
    created = ingest_if_needed(service, other, "outro")
    assert created.ingested is True
    assert service.has_document("outro")
