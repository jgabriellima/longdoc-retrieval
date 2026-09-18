"""
Store module for the retrieval API.
"""
import os
from collections.abc import Mapping
from pathlib import Path

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.indexes.db import connect

DEFAULT_DB_RELATIVE = Path(".longdoc") / "index.db"
ENV_DB = "LONGDOC_DB"


def resolve_db_path(
    cli_db: str | None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path:
    environment = os.environ if env is None else env
    working = cwd if cwd is not None else Path.cwd()
    if cli_db:
        return Path(cli_db).expanduser().resolve()
    env_db = environment.get(ENV_DB)
    if env_db:
        return Path(env_db).expanduser().resolve()
    return (working / DEFAULT_DB_RELATIVE).resolve()


def open_service(db_path: Path) -> RetrievalService:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return RetrievalService(connect(str(db_path)))


def resolve_document_id(service: RetrievalService, explicit_id: str | None) -> str:
    if explicit_id:
        if not service.has_document(explicit_id):
            raise UserError(f"document not found: {explicit_id}")
        return explicit_id
    ids = service.list_document_ids()
    if len(ids) == 0:
        raise UserError("no documents in index. run: longdoc ingest FILE")
    if len(ids) > 1:
        raise UserError(
            f"{len(ids)} documents in index. pass --id or FILE. run: longdoc list"
        )
    return ids[0]
