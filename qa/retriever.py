"""Find the chunks most relevant to a question using BM25.

BM25 is a classic keyword-ranking formula used by search engines. It is
written out by hand here (no extra libraries) so you can read exactly how
it scores each chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunking import Chunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "who", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words and numbers, and drop very common words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    """Rank chunks for a query with the BM25 formula."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_counts = [Counter(tokens) for tokens in self.doc_tokens]

        # Document frequency: in how many chunks does each word appear?
        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))
        n = len(chunks)
        # Rare words get a high IDF weight; words found everywhere get a low one.
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def score(self, query: str, index: int) -> float:
        """BM25 score of one chunk for the query."""
        counts = self.term_counts[index]
        length_ratio = self.doc_lengths[index] / self.avg_length
        total = 0.0
        for term in tokenize(query):
            if term not in counts:
                continue
            tf = counts[term]
            total += self.idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * length_ratio))
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks with their scores, best first."""
        scored = [(self.chunks[i], self.score(query, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
