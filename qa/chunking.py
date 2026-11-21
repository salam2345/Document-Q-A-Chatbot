"""Split documents into chunks that can be searched and passed to the LLM.

Two strategies are provided so their effect can be measured:

* fixed_size_chunks: cut the raw text every N words, ignoring structure.
* section_chunks: split on Markdown headings first, then into overlapping
  windows, and prefix each chunk with its document and section title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, plus where it came from."""

    text: str
    source: str  # file name, e.g. "delivery.md"
    section: str  # heading the text sits under ("" if none)


def load_documents(folder: str | Path) -> dict[str, str]:
    """Read every .md and .txt file in a folder. Returns {file name: text}."""
    folder = Path(folder)
    docs = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".md", ".txt"}:
            docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"No .md or .txt documents found in {folder}")
    return docs


def _word_windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Slide a window of `size` words across the list, stepping by size - overlap."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than the chunk size")
    if not words:
        return []
    step = size - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + size])
        if start + size >= len(words):
            break
    return windows


def fixed_size_chunks(docs: dict[str, str], size: int = 300) -> list[Chunk]:
    """Baseline: cut each document every `size` words with no overlap."""
    chunks = []
    for name, text in docs.items():
        for window in _word_windows(text.split(), size, 0):
            chunks.append(Chunk(text=" ".join(window), source=name, section=""))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown text into (heading, body) pairs.

    Text before the first heading gets the heading "".
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*)", line)
        if match:
            if any(part.strip() for part in body):
                sections.append((heading, "\n".join(body).strip()))
            heading = match.group(1).strip()
            body = []
        else:
            body.append(line)
    if any(part.strip() for part in body):
        sections.append((heading, "\n".join(body).strip()))
    return sections


def section_chunks(docs: dict[str, str], size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Improved: split by heading, then into overlapping windows.

    Each chunk starts with "Document title > Section" so that both the
    retriever and the LLM know what the text is about.
    """
    chunks = []
    for name, text in docs.items():
        title_match = re.search(r"^#\s+(.*)", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else name
        for heading, body in _split_sections(text):
            if heading == title:
                heading = ""
            label = f"{title} > {heading}" if heading else title
            for window in _word_windows(body.split(), size, overlap):
                chunks.append(Chunk(text=f"[{label}] " + " ".join(window), source=name, section=heading))
    return chunks
