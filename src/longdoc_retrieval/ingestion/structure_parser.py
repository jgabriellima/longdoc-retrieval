"""
Structure parser for the retrieval API.
"""
import re
from dataclasses import dataclass, field

from longdoc_retrieval.tokenize import approx_token_count

BLOCK_FALLBACK_THRESHOLD = 6000
BLOCK_TARGET_TOKENS = 2500

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")

_ATX_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_KEYWORD_SECTION_RE = re.compile(
    r"^(ARTICLE|SECTION|ARTIGO|SE[ÇC][ÃA]O|CAP[ÍI]TULO)\s+([IVXLCDM]+|\d+)\b[.:]?\s*(.*)$",
    re.IGNORECASE,
)
_KEYWORD_ANNEX_RE = re.compile(
    r"^(EXHIBIT|SCHEDULE|ANNEX|APPENDIX|ANEXO|AP[ÊE]NDICE)\s+\w+", re.IGNORECASE
)
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]\s*(.*)$")
_EMBEDDED_SENTENCE_RE = re.compile(r"[.!?]\s+\S")

_HEADING_CANDIDATE_MAX_LEN = 100
_NUMBERED_REMAINDER_MAX_WORDS = 10
_ALLCAPS_MAX_LEN = 80
_ALLCAPS_MAX_WORDS = 8
_ALLCAPS_MIN_ALPHA = 2


def _is_single_sentence(remainder: str) -> bool:
    return _EMBEDDED_SENTENCE_RE.search(remainder) is None


@dataclass
class Paragraph:
    start_offset: int
    end_offset: int
    text: str


@dataclass
class ParsedNode:
    title: str | None
    depth: int
    start_offset: int
    end_offset: int
    children: list["ParsedNode"] = field(default_factory=list)
    synthetic: bool = False


def split_paragraphs(text: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    pos = 0
    breaks = [m.span() for m in _BLANK_LINE_RE.finditer(text)]
    boundaries = [pos]
    for start, end in breaks:
        boundaries.append(start)
        boundaries.append(end)
    boundaries.append(len(text))

    for i in range(0, len(boundaries), 2):
        raw_start, raw_end = boundaries[i], boundaries[i + 1]
        if raw_start >= raw_end:
            continue
        chunk = text[raw_start:raw_end]
        lstripped = chunk.lstrip()
        lead = len(chunk) - len(lstripped)
        stripped = lstripped.rstrip()
        if not stripped:
            continue
        start = raw_start + lead
        end = start + len(stripped)
        paragraphs.append(Paragraph(start_offset=start, end_offset=end, text=stripped))

    return paragraphs


def _classify_heading(line: str) -> tuple[int, str] | None:
    m = _ATX_RE.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()

    m = _KEYWORD_SECTION_RE.match(line)
    if m and _is_single_sentence(m.group(3)):
        return 1, line.strip()

    if _KEYWORD_ANNEX_RE.match(line):
        return 1, line.strip()

    m = _NUMBERED_RE.match(line)
    if m:
        number, remainder = m.group(1), m.group(2).strip()
        words = remainder.split()
        if len(words) <= _NUMBERED_REMAINDER_MAX_WORDS and _is_single_sentence(remainder):
            depth = number.count(".") + 1
            return depth, line.strip()

    if len(line) <= _ALLCAPS_MAX_LEN and len(line.split()) <= _ALLCAPS_MAX_WORDS:
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count >= _ALLCAPS_MIN_ALPHA and line == line.upper() and line != line.lower():
            return 1, line.strip()

    return None


def _detect_headings(paragraphs: list[Paragraph]) -> list[tuple[int, str, Paragraph]]:
    headings = []
    for para in paragraphs:
        if "\n" in para.text or len(para.text) > _HEADING_CANDIDATE_MAX_LEN:
            continue
        result = _classify_heading(para.text)
        if result is not None:
            depth, title = result
            headings.append((depth, title, para))
    return headings


def _build_heading_tree(
    headings: list[tuple[int, str, Paragraph]], doc_end: int
) -> ParsedNode:
    root = ParsedNode(title=None, depth=0, start_offset=0, end_offset=doc_end)
    if not headings:
        return root

    stack: list[ParsedNode] = [root]

    for idx, (depth, title, para) in enumerate(headings):
        next_start = headings[idx + 1][2].start_offset if idx + 1 < len(headings) else doc_end
        node = ParsedNode(
            title=title,
            depth=depth,
            start_offset=para.start_offset,
            end_offset=next_start,
        )
        while len(stack) > 1 and stack[-1].depth >= depth:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)

    _fix_parent_ends(root)
    _tile_children(root)
    return root


def _fix_parent_ends(node: ParsedNode) -> None:
    for child in node.children:
        _fix_parent_ends(child)
    if node.children and node.depth > 0:
        node.end_offset = node.children[-1].end_offset


def _tile_children(node: ParsedNode) -> None:
    if not node.children:
        return

    tiled: list[ParsedNode] = []
    cursor = node.start_offset
    for child in node.children:
        if child.start_offset > cursor:
            tiled.append(
                ParsedNode(
                    title=None,
                    depth=child.depth,
                    start_offset=cursor,
                    end_offset=child.start_offset,
                    synthetic=True,
                )
            )
        tiled.append(child)
        cursor = child.end_offset
    if cursor < node.end_offset:
        tiled.append(
            ParsedNode(
                title=None,
                depth=tiled[-1].depth if tiled else node.depth + 1,
                start_offset=cursor,
                end_offset=node.end_offset,
                synthetic=True,
            )
        )

    node.children = tiled
    for child in node.children:
        _tile_children(child)


def _apply_block_fallback(node: ParsedNode, text: str) -> None:
    if node.children:
        for child in node.children:
            _apply_block_fallback(child, text)
        return

    span_text = text[node.start_offset : node.end_offset]
    if approx_token_count(span_text) <= BLOCK_FALLBACK_THRESHOLD:
        return

    paragraphs = split_paragraphs(span_text)
    if len(paragraphs) <= 1:
        return

    blocks: list[ParsedNode] = []
    block_start = node.start_offset
    block_tokens = 0
    block_index = 1

    def _flush(end_offset: int) -> None:
        nonlocal block_start, block_tokens, block_index
        blocks.append(
            ParsedNode(
                title=f"Bloco {block_index}",
                depth=node.depth + 1,
                start_offset=block_start,
                end_offset=end_offset,
                synthetic=True,
            )
        )
        block_index += 1
        block_start = end_offset
        block_tokens = 0

    for para in paragraphs:
        abs_start = node.start_offset + para.start_offset
        para_tokens = approx_token_count(para.text)
        if block_tokens > 0 and block_tokens + para_tokens > BLOCK_TARGET_TOKENS:
            _flush(abs_start)
        block_tokens += para_tokens

    _flush(node.end_offset)
    node.children = blocks


def parse_structure(text: str) -> ParsedNode:
    paragraphs = split_paragraphs(text)
    headings = _detect_headings(paragraphs)
    root = _build_heading_tree(headings, len(text))
    _apply_block_fallback(root, text)
    return root
