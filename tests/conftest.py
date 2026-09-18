from pathlib import Path

import pytest

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.domain.document import Document, DocumentMetadata
from longdoc_retrieval.indexes.db import connect

SAMPLE_TEXT = """# Convênio 169

Preâmbulo do instrumento.

## ARTIGO 1 - Objeto

O objeto deste convênio é o repasse de recursos no valor de R$ 150.000,00.

## ARTIGO 2 - Vigência

A vigência inicia em 01/03/2022 e termina em 31/12/2023.
"""


@pytest.fixture
def conn():
    connection = connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def service(conn) -> RetrievalService:
    return RetrievalService(conn)


@pytest.fixture
def ingested(service) -> RetrievalService:
    service.ingest(
        Document(
            document_id="convenio",
            content=SAMPLE_TEXT,
            metadata=DocumentMetadata(source="convenio.md"),
        )
    )
    return service


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "convenio.md"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return path
