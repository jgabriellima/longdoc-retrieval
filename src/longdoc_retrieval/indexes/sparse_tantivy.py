import re

import tantivy

from longdoc_retrieval.indexes.sparse import SparseHit
from longdoc_retrieval.ingestion.node_builder import RetrievalUnit

_CONTENT_TOKENIZER_NAME = "pt_stem"


def _content_analyzer() -> "tantivy.TextAnalyzer":
    return (
        tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.lowercase())
        .filter(tantivy.Filter.stemmer("portuguese"))
        .build()
    )


def build_index() -> tantivy.Index:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("content", stored=True, tokenizer_name=_CONTENT_TOKENIZER_NAME)
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
    return _WHITESPACE_RE.sub(" ", fragment).strip()
