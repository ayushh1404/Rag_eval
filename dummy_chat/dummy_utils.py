import re
import uuid
from typing import List

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def generate_public_id() -> str:
    return str(uuid.uuid4())


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n\n".join(lines)


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - CHUNK_OVERLAP

    return chunks
