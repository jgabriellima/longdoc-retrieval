_BOM = "﻿"


def normalize(text: str) -> str:
    text = text.removeprefix(_BOM)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text
