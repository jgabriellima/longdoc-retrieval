"""Console entry: parse argv, print, map errors to exit codes."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from longdoc_retrieval.app.ask import AskRunner, require_api_key, run_ask
from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.app.ingest import document_id_from_path, ingest_file, ingest_if_needed
from longdoc_retrieval.app.store import open_service, resolve_db_path, resolve_document_id
from longdoc_retrieval.cli.parser import build_parser
from longdoc_retrieval.cli.render import (
    emit_candidates,
    emit_documents,
    emit_json,
    emit_outline,
    emit_package,
    emit_read,
)
from longdoc_retrieval.config import RetrievalConfig


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _parse_ask_args(args: list[str]) -> tuple[Path | None, str]:
    if len(args) >= 2:
        candidate = Path(args[0])
        if candidate.is_file():
            return candidate, " ".join(args[1:])
    return None, " ".join(args)


async def _dispatch(args: argparse.Namespace, ask_runner: AskRunner | None) -> None:
    service = open_service(resolve_db_path(args.db))
    as_json = args.json

    if args.command == "ingest":
        path = Path(args.file)
        document_id = document_id_from_path(path, args.document_id)
        result = ingest_file(service, path, document_id)
        if result.replaced:
            print(f"replaced {result.document_id}", file=sys.stderr)
        if as_json:
            emit_json({"document_id": result.document_id, "replaced": result.replaced})
        else:
            print(result.document_id)
        return

    if args.command == "list":
        records = service.list_documents()
        if as_json:
            emit_json(records)
        else:
            emit_documents(records)
        return

    if args.command == "rm":
        if not service.has_document(args.document_id):
            raise UserError(f"document not found: {args.document_id}")
        service.delete_document(args.document_id)
        if as_json:
            emit_json({"deleted": args.document_id})
        else:
            print(args.document_id)
        return

    if args.command == "outline":
        document_id = resolve_document_id(service, args.document_id)
        outline = await service.document_outline(document_id, depth=args.depth)
        emit_json(outline) if as_json else emit_outline(outline)
        return

    if args.command == "search":
        document_id = resolve_document_id(service, args.document_id)
        hits = await service.search(document_id, args.query, limit=args.limit)
        emit_json(hits) if as_json else emit_candidates(hits)
        return

    if args.command == "find-exact":
        document_id = resolve_document_id(service, args.document_id)
        hits = await service.find_exact(document_id, args.expression, limit=args.limit)
        emit_json(hits) if as_json else emit_candidates(hits)
        return

    if args.command == "read":
        has_node = args.node is not None
        has_start = args.start is not None
        has_end = args.end is not None
        if has_node and (has_start or has_end):
            raise UserError("do not pass NODE and --start/--end together")
        if has_start != has_end:
            raise UserError("read range requires both --start and --end")
        if not has_node and not (has_start and has_end):
            raise UserError("read requires NODE or --start and --end")
        document_id = resolve_document_id(service, args.document_id)
        if has_node:
            read_result = await service.read_node(document_id, args.node)
        else:
            read_result = await service.read_range(document_id, args.start, args.end)
        emit_json(read_result) if as_json else emit_read(read_result)
        return

    if args.command == "ask":
        require_api_key()
        file_path, question = _parse_ask_args(args.args)
        if file_path is not None:
            document_id = document_id_from_path(file_path, args.document_id)
            ingest_if_needed(service, file_path, document_id)
        else:
            document_id = resolve_document_id(service, args.document_id)
        package = await run_ask(
            service,
            document_id,
            question,
            runner=ask_runner,
            config=RetrievalConfig.from_env(),
        )
        emit_json(package) if as_json else emit_package(package)
        return

    raise UserError(f"unknown command: {args.command}")


def main(
    argv: list[str] | None = None,
    *,
    ask_runner: AskRunner | None = None,
    load_env: bool = True,
) -> int:
    if load_env:
        _load_dotenv()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in (0, None):
            return 0
        return int(code) if isinstance(code, int) else 2

    try:
        asyncio.run(_dispatch(args, ask_runner))
        return 0
    except UserError as exc:
        print(exc.message, file=sys.stderr)
        return exc.exit_code
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI process boundary
        print(f"internal error: {exc}", file=sys.stderr)
        return 1


def entrypoint() -> None:
    raise SystemExit(main())
