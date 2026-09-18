import json
from pathlib import Path

import pytest

from longdoc_retrieval.cli import main
from longdoc_retrieval.domain.metrics import RetrievalMetrics
from longdoc_retrieval.domain.package import EvidencePackage
from tests.conftest import SAMPLE_TEXT


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]):
    code = main(["--help"])
    assert code == 0
    assert "ingest" in capsys.readouterr().out


def test_ingest_list_outline_search_read_rm(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    doc = tmp_path / "convenio.md"
    doc.write_text(SAMPLE_TEXT, encoding="utf-8")
    db = tmp_path / "index.db"

    assert main(["--db", str(db), "ingest", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "convenio" in out

    assert main(["--db", str(db), "--json", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["document_id"] == "convenio"

    assert main(["--db", str(db), "outline"]) == 0
    outline = capsys.readouterr().out
    assert "ARTIGO" in outline or "Convênio" in outline or "convenio#" in outline

    assert main(["--db", str(db), "search", "vigência"]) == 0
    search_out = capsys.readouterr().out
    assert "vigência" in search_out.lower() or "2022" in search_out or "ARTIGO" in search_out

    assert main(["--db", str(db), "find-exact", "150.000"]) == 0
    exact_out = capsys.readouterr().out
    assert "150.000" in exact_out

    assert main(["--db", str(db), "read", "convenio#000000"]) == 0
    read_out = capsys.readouterr().out
    assert "Convênio" in read_out or "convênio" in read_out.lower() or len(read_out) > 0

    assert main(["--db", str(db), "rm", "convenio"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "--json", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_ambiguous_document_is_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db = tmp_path / "index.db"
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\nOne.\n", encoding="utf-8")
    b.write_text("# B\n\nTwo.\n", encoding="utf-8")
    assert main(["--db", str(db), "ingest", str(a)]) == 0
    assert main(["--db", str(db), "ingest", str(b)]) == 0
    capsys.readouterr()
    code = main(["--db", str(db), "outline"])
    err = capsys.readouterr().err
    assert code == 2
    assert "2 documents" in err


def test_empty_list_is_exit_zero_empty_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db = tmp_path / "empty.db"
    assert main(["--db", str(db), "--json", "list"]) == 0
    assert capsys.readouterr().out.strip() == "[]"


def test_read_rejects_node_and_offsets_together(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    doc = tmp_path / "convenio.md"
    doc.write_text(SAMPLE_TEXT, encoding="utf-8")
    db = tmp_path / "index.db"
    assert main(["--db", str(db), "ingest", str(doc)]) == 0
    capsys.readouterr()
    code = main(["--db", str(db), "read", "convenio#000000", "--start", "0", "--end", "10"])
    assert code == 2
    assert "together" in capsys.readouterr().err


def test_ask_without_api_key_is_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    doc = tmp_path / "convenio.md"
    doc.write_text(SAMPLE_TEXT, encoding="utf-8")
    db = tmp_path / "index.db"
    assert main(["--db", str(db), "ingest", str(doc)], load_env=False) == 0
    capsys.readouterr()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["--db", str(db), "ask", "qual o valor?"], load_env=False)
    assert code == 2
    assert "API key" in capsys.readouterr().err


def test_ask_with_injected_runner_and_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    doc = tmp_path / "convenio.md"
    doc.write_text(SAMPLE_TEXT, encoding="utf-8")
    db = tmp_path / "index.db"

    async def fake_runner(service, document_id, question, config=None, llm_clients=None):
        return EvidencePackage(
            request_id="req",
            document_id=document_id,
            question=question,
            status="sufficient",
            retrieval_summary="found",
            answer="R$ 150.000,00",
            metrics=RetrievalMetrics(
                document_id=document_id,
                document_tokens=10,
                question_tokens=4,
            ),
        )

    code = main(
        ["--db", str(db), "--json", "ask", str(doc), "qual o valor?"],
        ask_runner=fake_runner,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "sufficient"
    assert payload["answer"] == "R$ 150.000,00"
    assert payload["document_id"] == "convenio"


def test_pdf_ingest_is_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    db = tmp_path / "index.db"
    code = main(["--db", str(db), "ingest", str(pdf)])
    assert code == 2
    assert "PDF" in capsys.readouterr().err
