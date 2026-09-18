"""
Parser for the retrieval CLI.
"""
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longdoc",
        description="Ingest long documents and retrieve evidence without embeddings.",
    )
    parser.add_argument("--db", help="SQLite index path (default: ./.longdoc/index.db)")
    parser.add_argument("--json", action="store_true", help="machine-readable stdout")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a UTF-8 text document")
    ingest.add_argument("file")
    ingest.add_argument("--id", dest="document_id")

    sub.add_parser("list", help="list indexed documents")

    rm = sub.add_parser("rm", help="delete a document from the index")
    rm.add_argument("document_id")

    outline = sub.add_parser("outline", help="print the section tree")
    outline.add_argument("--id", dest="document_id")
    outline.add_argument("--depth", type=int, default=2)

    search = sub.add_parser("search", help="BM25 sparse search")
    search.add_argument("query")
    search.add_argument("--id", dest="document_id")
    search.add_argument("--limit", type=int, default=20)

    exact = sub.add_parser("find-exact", help="identifier / regex scan")
    exact.add_argument("expression")
    exact.add_argument("--id", dest="document_id")
    exact.add_argument("--limit", type=int, default=20)

    read = sub.add_parser("read", help="read a node or an offset range")
    read.add_argument("node", nargs="?")
    read.add_argument("--id", dest="document_id")
    read.add_argument("--start", type=int)
    read.add_argument("--end", type=int)

    ask = sub.add_parser("ask", help="agentic retrieval loop")
    ask.add_argument("args", nargs="+", help="QUESTION, or FILE QUESTION")
    ask.add_argument("--id", dest="document_id")
    return parser
