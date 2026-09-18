from longdoc_retrieval.domain.document import Document, DocumentMetadata
from longdoc_retrieval.indexes import sparse, structural
from tests.conftest import SAMPLE_TEXT


def test_list_documents_empty(conn):
    assert structural.list_documents(conn) == []


def test_list_documents_returns_id_source_and_tokens(ingested, conn):
    records = structural.list_documents(conn)
    assert len(records) == 1
    assert records[0].document_id == "convenio"
    assert records[0].source == "convenio.md"
    assert records[0].token_count > 0


def test_document_exists(ingested, conn):
    assert structural.document_exists(conn, "convenio") is True
    assert structural.document_exists(conn, "missing") is False


def test_delete_document_removes_structural_and_fts_rows(ingested, conn):
    structural.delete_document(conn, "convenio")
    sparse.delete_document(conn, "convenio")

    assert structural.list_documents(conn) == []
    assert structural.document_exists(conn, "convenio") is False
    hits = sparse.search(conn, "convenio", "vigência", limit=5)
    assert hits == []


def test_reingest_does_not_leave_orphan_nodes(service, conn):
    service.ingest(Document(document_id="doc", content=SAMPLE_TEXT))
    nodes_before = structural.get_all_nodes(conn, "doc")
    assert len(nodes_before) > 1

    service.delete_document("doc")
    service.ingest(Document(document_id="doc", content="# Only title\n\nShort.\n"))
    nodes_after = structural.get_all_nodes(conn, "doc")
    assert all(n.document_id == "doc" for n in nodes_after)
    assert len(nodes_after) < len(nodes_before)


def test_delete_missing_document_raises(service):
    try:
        service.delete_document("ghost")
    except KeyError as exc:
        assert "ghost" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_service_list_and_has_document(ingested):
    records = ingested.list_documents()
    assert [r.document_id for r in records] == ["convenio"]
    assert ingested.has_document("convenio") is True
    assert ingested.has_document("other") is False


def test_service_ingest_replaces_after_delete(service):
    service.ingest(
        Document(
            document_id="x",
            content=SAMPLE_TEXT,
            metadata=DocumentMetadata(source="a.md"),
        )
    )
    service.delete_document("x")
    service.ingest(
        Document(
            document_id="x",
            content="# Other\n\nBody.\n",
            metadata=DocumentMetadata(source="b.md"),
        )
    )
    listed = service.list_documents()
    assert len(listed) == 1
    assert listed[0].source == "b.md"
