import json
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from longdoc_retrieval.api.schemas import DocumentOutline, DocumentRecord, OutlineNode
from longdoc_retrieval.domain.package import EvidencePackage
from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.retrieval.reader import ReadResult


def emit_json(value: Any) -> None:
    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump()
    elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
        payload = [item.model_dump() for item in value]
    else:
        payload = value
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    sys.stdout.write("\n")


def emit_documents(records: Sequence[DocumentRecord]) -> None:
    for record in records:
        source = record.source or "-"
        sys.stdout.write(f"{record.document_id}\tsource={source}\ttokens={record.token_count}\n")


def emit_outline(outline: DocumentOutline) -> None:
    def walk(node: OutlineNode, indent: int) -> None:
        title = node.title or "(untitled)"
        sys.stdout.write(f"{'  ' * indent}{node.node_id}  {title}  ({node.token_count} tokens)\n")
        for child in node.children:
            walk(child, indent + 1)

    walk(outline.root, 0)


def emit_candidates(candidates: Sequence[RetrievalCandidate]) -> None:
    if not candidates:
        return
    for candidate in candidates:
        score = "" if candidate.lexical_score is None else f"{candidate.lexical_score:.3f}"
        score_part = f"{score}\t" if score else ""
        sys.stdout.write(
            f"{score_part}{candidate.node_id}\t"
            f"{candidate.start_offset}-{candidate.end_offset}\n"
        )
        preview = candidate.preview.replace("\n", " ").strip()
        if preview:
            sys.stdout.write(f"  {preview}\n")


def emit_read(result: ReadResult) -> None:
    sys.stdout.write(result.text)
    if not result.text.endswith("\n"):
        sys.stdout.write("\n")
    if result.truncated:
        sys.stdout.write(
            f"# truncated at {result.end_offset}"
            + (
                f" continue={result.continuation_offset}"
                if result.continuation_offset is not None
                else ""
            )
            + "\n"
        )


def emit_package(package: EvidencePackage) -> None:
    sys.stdout.write(f"status: {package.status}\n")
    if package.answer:
        sys.stdout.write(f"answer: {package.answer}\n")
    else:
        sys.stdout.write("answer: (none)\n")
    if package.evidence:
        sys.stdout.write("\nevidence:\n")
        for item in package.evidence:
            sys.stdout.write(
                f"  {item.evidence_id}\t{item.start_offset}-{item.end_offset}\n"
            )
    if package.unresolved_questions:
        sys.stdout.write("\nunresolved:\n")
        for question in package.unresolved_questions:
            sys.stdout.write(f"  - {question}\n")
