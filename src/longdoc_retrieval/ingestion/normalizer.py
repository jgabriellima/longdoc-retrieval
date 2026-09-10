"""Minimal, deterministic text normalization.

Only CRLF/CR -> LF and BOM stripping. Nothing else - no whitespace
collapsing, no trimming of interior content. Offsets are canonical: every
offset produced anywhere downstream is relative to this normalized text,
and it must be possible to round-trip `content[start_offset:end_offset]`
reliably, which rules out any normalization step that isn't a
straightforward character substitution.
"""

_BOM = "﻿"


def normalize(text: str) -> str:
    text = text.removeprefix(_BOM)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text
