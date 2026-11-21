"""Put the pieces together: chunk the documents, retrieve, prompt the LLM.

CONFIGS holds the two versions that evaluate.py compares. Change a value
here (or add a new config) and re-run the evaluation to see its effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, fixed_size_chunks, load_documents, section_chunks
from .retriever import BM25Retriever

BASELINE_PROMPT = "You are a helpful assistant for the Kikapu grocery delivery app. Answer the user's question."

IMPROVED_PROMPT = """You are the help-center assistant for Kikapu, a grocery delivery app.
Rules:
1. Answer ONLY from the context provided. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "I don't know based on the Kikapu help documents."
3. Keep answers short (1-3 sentences) and include exact numbers, prices and time limits from the context.
4. If the answer differs between Kamloops and Nairobi, give both."""


@dataclass(frozen=True)
class Config:
    name: str
    chunking: str  # "fixed" or "section"
    chunk_size: int
    overlap: int
    top_k: int
    system_prompt: str


CONFIGS = {
    "baseline": Config("baseline", chunking="fixed", chunk_size=300, overlap=0, top_k=1, system_prompt=BASELINE_PROMPT),
    "improved": Config("improved", chunking="section", chunk_size=120, overlap=30, top_k=4, system_prompt=IMPROVED_PROMPT),
}

DOCS_FOLDER = Path(__file__).resolve().parent.parent / "docs"


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Chunk]


class QAPipeline:
    """Answers questions about a folder of documents."""

    def __init__(self, config: Config, llm, docs_folder: str | Path = DOCS_FOLDER):
        self.config = config
        self.llm = llm
        docs = load_documents(docs_folder)
        if config.chunking == "fixed":
            self.chunks = fixed_size_chunks(docs, size=config.chunk_size)
        elif config.chunking == "section":
            self.chunks = section_chunks(docs, size=config.chunk_size, overlap=config.overlap)
        else:
            raise ValueError(f"Unknown chunking strategy {config.chunking!r}")
        self.retriever = BM25Retriever(self.chunks)

    def build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(chunk.text for chunk in chunks)
        return f"Context:\n{context}\n\nQuestion: {question}"

    def ask(self, question: str) -> Answer:
        results = self.retriever.search(question, top_k=self.config.top_k)
        chunks = [chunk for chunk, _score in results]
        reply = self.llm.complete(self.config.system_prompt, self.build_prompt(question, chunks))
        return Answer(question=question, answer=reply, sources=chunks)
