"""Tantivy sparse index - a second implementation of the sparse-retrieval
backend (see `indexes/sparse.py`'s docstring), meant to be A/B'd against
SQLite FTS5 on real data rather than replace it outright - both
implementations are kept side by side, selected at `RetrievalService`
construction time.

Motivation, verified empirically (not assumed): SQLite FTS5's `unicode61`
tokenizer has no PT-BR stemming, so a query for "valor total" never matches
a clause that only says "totaliza". A disposable probe script (schema +
`Filter.stemmer("portuguese")` + a query for "valor total" against a corpus
containing only "totaliza") returned a positive BM25 match, confirming
Tantivy's Portuguese Snowball stemmer closes that exact gap.

One index per process (module-level, keyed by nothing - `document_id` is a
stored+filtered field within the single index, mirroring the SQLite
FTS5 table's `document_id UNINDEXED` + `WHERE document_id = ?` pattern),
since `RetrievalUnit`/`Document` content already carries `document_id`.
"""

import re

import tantivy

from longdoc_retrieval.indexes.sparse import SparseHit
from longdoc_retrieval.ingestion.node_builder import RetrievalUnit

_CONTENT_TOKENIZER_NAME = "pt_stem"

# `parse_query`'s `conjunction_by_default` defaults to False (disjunction/OR
# of terms) - verified via probe script, matching `indexes/sparse.py`'s own
# `_build_match_query` OR-of-terms behavior, so switching backends doesn't
# also silently change match semantics from OR to AND.


def _content_analyzer() -> "tantivy.TextAnalyzer":
    return (
        tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.lowercase())
        .filter(tantivy.Filter.stemmer("portuguese"))
        .build()
    )


def build_index() -> tantivy.Index:
    """In-memory index (no directory path) - same lifetime/scope as the
    in-memory SQLite connection this backend is compared against. A
    persistent, on-disk index is a production deployment concern, not
    something this module needs to decide.
    """

    builder = tantivy.SchemaBuilder()
    builder.add_text_field("content", stored=True, tokenizer_name=_CONTENT_TOKENIZER_NAME)
    # "raw" tokenizer_name stores these fields verbatim/untokenized, so
    # `Query.term_query` does exact-match filtering on them (document_id
    # scoping, and unit/node identity on the way back out) - mirrors the
    # SQLite schema's `unit_id UNINDEXED, document_id UNINDEXED` columns.
    builder.add_text_field("unit_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("document_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("node_id", stored=True, tokenizer_name="raw")
    builder.add_integer_field("start_offset", stored=True)
    builder.add_integer_field("end_offset", stored=True)
    schema = builder.build()

    index = tantivy.Index(schema)
    index.register_tokenizer(_CONTENT_TOKENIZER_NAME, _content_analyzer())
    return index


def index_units(index: tantivy.Index, document_text: str, units: list[RetrievalUnit]) -> None:
    writer = index.writer()
    for unit in units:
        doc = tantivy.Document()
        doc.add_text("content", document_text[unit.start_offset : unit.end_offset])
        doc.add_text("unit_id", unit.unit_id)
        doc.add_text("document_id", unit.document_id)
        doc.add_text("node_id", unit.node_id)
        doc.add_integer("start_offset", unit.start_offset)
        doc.add_integer("end_offset", unit.end_offset)
        writer.add_document(doc)
    writer.commit()
    index.reload()


def search(index: tantivy.Index, document_id: str, query: str, limit: int) -> list[SparseHit]:
    schema = index.schema
    document_id_query = tantivy.Query.term_query(schema, "document_id", document_id)

    try:
        text_query = index.parse_query(query, ["content"])
    except ValueError:
        # Malformed query-language syntax (e.g. stray `"`/`(`/`*`) - verified
        # via probe script to raise ValueError here, exactly where
        # `indexes/sparse.py::search` catches `sqlite3.OperationalError` for
        # the same reason: don't propagate a syntax error from what the
        # caller only intended as free-text.
        return []

    combined = tantivy.Query.boolean_query(
        [
            (tantivy.Occur.Must, document_id_query),
            (tantivy.Occur.Must, text_query),
        ]
    )

    searcher = index.searcher()
    result = searcher.search(combined, limit=limit)
    if not result.hits:
        return []

    # Tantivy's default search order is descending or higher-is-better (spec
    # confirmed via probe script and `Searcher.search`'s own docstring) -
    # unlike SQLite's raw ascending `bm25()`, this score is used as-is, not
    # negated.
    snippet_generator = tantivy.SnippetGenerator.create(searcher, text_query, schema, "content")

    hits = []
    for score, address in result.hits:
        doc = searcher.doc(address)
        snippet = snippet_generator.snippet_from_doc(doc)
        hits.append(
            SparseHit(
                unit_id=doc["unit_id"][0],
                node_id=doc["node_id"][0],
                start_offset=doc["start_offset"][0],
                end_offset=doc["end_offset"][0],
                score=score,
                snippet=_normalize_snippet(snippet.fragment()),
            )
        )
    return hits


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_snippet(fragment: str) -> str:
    # `Snippet.fragment()` is already plain text (no HTML, unlike
    # `.to_html()`), but preserves the source's own whitespace/newlines
    # verbatim; collapsing it keeps `SparseHit.snippet` comparable in shape
    # to SQLite's single-line `snippet()` preview.
    return _WHITESPACE_RE.sub(" ", fragment).strip()
