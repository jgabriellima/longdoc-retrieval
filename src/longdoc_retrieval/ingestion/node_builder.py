"""
Node builder for the retrieval API.
"""
import re

from pydantic import BaseModel

from longdoc_retrieval.domain.node import DocumentNode
from longdoc_retrieval.ingestion.structure_parser import ParsedNode, split_paragraphs
from longdoc_retrieval.tokenize import approx_token_count

MIN_UNIT_TOKENS = 150
MAX_UNIT_TOKENS = 400
MAX_UNIT_TOKENS_HARD = 500


class RetrievalUnit(BaseModel):
    unit_id: str
    document_id: str
    node_id: str
    start_offset: int
    end_offset: int
    token_count: int
    ordinal: int
    truncated_split: bool = False


def build_nodes(document_id: str, text: str, root: ParsedNode) -> list[DocumentNode]:
    nodes: list[DocumentNode] = []
    counter = {"n": 0}

    def _visit(parsed: ParsedNode, parent_id: str | None) -> str:
        ordinal = counter["n"]
        counter["n"] += 1
        node_id = f"{document_id}#{ordinal:06d}"

        span_text = text[parsed.start_offset : parsed.end_offset]
        node = DocumentNode(
            node_id=node_id,
            document_id=document_id,
            parent_id=parent_id,
            children_ids=[],
            title=parsed.title,
            start_offset=parsed.start_offset,
            end_offset=parsed.end_offset,
            start_page=None,
            end_page=None,
            token_count=approx_token_count(span_text),
            depth=parsed.depth,
            metadata={"synthetic": True} if parsed.synthetic else {},
        )
        nodes.append(node)

        child_ids = [_visit(child, node_id) for child in parsed.children]
        node.children_ids = child_ids
        return node_id

    _visit(root, None)
    return nodes


def _leaves(nodes: list[DocumentNode]) -> list[DocumentNode]:
    return [n for n in nodes if not n.children_ids]


def _lowest_common_ancestor(
    nodes_by_id: dict[str, DocumentNode], node_id_a: str, node_id_b: str
) -> str:
    def ancestors(node_id: str) -> list[str]:
        chain = []
        current: str | None = node_id
        while current is not None:
            chain.append(current)
            current = nodes_by_id[current].parent_id
        return chain

    ancestors_a = ancestors(node_id_a)
    ancestors_b = set(ancestors(node_id_b))
    for candidate in ancestors_a:
        if candidate in ancestors_b:
            return candidate
    return node_id_a


def build_retrieval_units(
    document_id: str, text: str, nodes: list[DocumentNode]
) -> list[RetrievalUnit]:
    nodes_by_id = {n.node_id: n for n in nodes}
    leaves = sorted(_leaves(nodes), key=lambda n: n.start_offset)

    pieces: list[tuple[int, int, str, int, bool]] = []
    cursor = 0
    for leaf in leaves:
        span_text = text[leaf.start_offset : leaf.end_offset]
        paragraphs = split_paragraphs(span_text)
        if not paragraphs:
            continue
        for para in paragraphs:
            abs_start = max(leaf.start_offset + para.start_offset, cursor)
            abs_end = leaf.start_offset + para.end_offset
            if abs_start >= abs_end:
                continue
            cursor = abs_end
            para_tokens = approx_token_count(text[abs_start:abs_end])
            if para_tokens > MAX_UNIT_TOKENS_HARD:
                for s, e in _hard_slice(text, abs_start, abs_end):
                    pieces.append((s, e, leaf.node_id, approx_token_count(text[s:e]), True))
            else:
                pieces.append((abs_start, abs_end, leaf.node_id, para_tokens, False))

    units: list[RetrievalUnit] = []
    ordinal = 0
    i = 0
    while i < len(pieces):
        start, end, node_id, tokens, truncated = pieces[i]
        unit_leaf_ids = {node_id}
        unit_start, unit_end, unit_tokens = start, end, tokens
        unit_truncated = truncated
        j = i + 1
        while j < len(pieces):
            _, next_end, next_node_id, next_tokens, next_truncated = pieces[j]
            if unit_tokens + next_tokens > MAX_UNIT_TOKENS:
                break
            unit_end = next_end
            unit_tokens += next_tokens
            unit_leaf_ids.add(next_node_id)
            unit_truncated = unit_truncated or next_truncated
            j += 1

        resolved_node_id = _resolve_unit_node_id(nodes_by_id, unit_leaf_ids)
        units.append(
            RetrievalUnit(
                unit_id=f"{document_id}#unit-{ordinal:06d}",
                document_id=document_id,
                node_id=resolved_node_id,
                start_offset=unit_start,
                end_offset=unit_end,
                token_count=unit_tokens,
                ordinal=ordinal,
                truncated_split=unit_truncated,
            )
        )
        ordinal += 1
        i = j

    _merge_small_units(units, nodes_by_id)
    return units


def _resolve_unit_node_id(nodes_by_id: dict[str, DocumentNode], leaf_ids: set[str]) -> str:
    ids = list(leaf_ids)
    result = ids[0]
    for other in ids[1:]:
        result = _lowest_common_ancestor(nodes_by_id, result, other)
    return result


def _hard_slice(text: str, start: int, end: int) -> list[tuple[int, int]]:
    target_chars = start + int(
        (end - start) * (MAX_UNIT_TOKENS / max(approx_token_count(text[start:end]), 1))
    )
    target_chars = min(max(target_chars, start + 1), end - 1)

    window = text[start:end]
    rel_target = target_chars - start
    split_at = None
    for match in re.finditer(r"[.!?]\s+", window):
        if match.end() <= rel_target:
            split_at = match.end()
        else:
            break
    if split_at is None:
        ws = window.rfind(" ", 0, rel_target)
        split_at = ws + 1 if ws != -1 else rel_target

    split_at = max(1, min(split_at, len(window) - 1))
    mid = start + split_at

    if approx_token_count(text[mid:end]) > MAX_UNIT_TOKENS_HARD:
        rest = _hard_slice(text, mid, end)
    else:
        rest = [(mid, end)]
    return [(start, mid), *rest]


def _merge_small_units(units: list[RetrievalUnit], nodes_by_id: dict[str, DocumentNode]) -> None:
    i = 0
    while i < len(units) - 1:
        if units[i].token_count >= MIN_UNIT_TOKENS:
            i += 1
            continue
        small, nxt = units[i], units[i + 1]
        if small.token_count + nxt.token_count > MAX_UNIT_TOKENS_HARD:
            i += 1
            continue
        nxt.start_offset = small.start_offset
        nxt.token_count += small.token_count
        nxt.truncated_split = nxt.truncated_split or small.truncated_split
        nxt.node_id = _lowest_common_ancestor(nodes_by_id, small.node_id, nxt.node_id)
        units.pop(i)

    if len(units) >= 2 and units[-1].token_count < MIN_UNIT_TOKENS:
        prev, last = units[-2], units[-1]
        if prev.token_count + last.token_count <= MAX_UNIT_TOKENS_HARD:
            prev.end_offset = last.end_offset
            prev.token_count += last.token_count
            prev.truncated_split = prev.truncated_split or last.truncated_split
            prev.node_id = _lowest_common_ancestor(nodes_by_id, prev.node_id, last.node_id)
            units.pop()
