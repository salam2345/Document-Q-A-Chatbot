import pytest

from qa.chunking import Chunk
from qa.retriever import BM25Retriever, tokenize

CHUNKS = [
    Chunk("The delivery fee is $4.99 in Kamloops.", "delivery.md", "Fees"),
    Chunk("Refunds to your card take 5 to 7 business days.", "refunds.md", "Refunds"),
    Chunk("Reset links for your password expire after 30 minutes.", "accounts.md", "Password"),
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Delivery fee?") == ["delivery", "fee"]


def test_search_ranks_the_relevant_chunk_first():
    retriever = BM25Retriever(CHUNKS)
    best, score = retriever.search("how long do card refunds take", top_k=1)[0]
    assert best.source == "refunds.md"
    assert score > 0


def test_search_returns_top_k_results_in_order():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("password reset", top_k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_empty_chunk_list_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])
