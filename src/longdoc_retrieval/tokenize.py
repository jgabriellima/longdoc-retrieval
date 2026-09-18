import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def approx_token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))
