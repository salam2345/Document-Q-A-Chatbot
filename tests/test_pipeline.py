"""End-to-end checks that run offline with the mock LLM."""

from qa.llm import MockLLM
from qa.pipeline import CONFIGS, QAPipeline


def test_improved_pipeline_retrieves_the_right_document():
    pipeline = QAPipeline(CONFIGS["improved"], MockLLM())
    result = pipeline.ask("How long is a password reset link valid?")
    assert "accounts_and_app.md" in [chunk.source for chunk in result.sources]
    assert len(result.sources) == CONFIGS["improved"].top_k


def test_prompt_contains_context_and_question():
    pipeline = QAPipeline(CONFIGS["improved"], MockLLM())
    chunks = pipeline.retriever.search("delivery fee", top_k=2)
    prompt = pipeline.build_prompt("What is the delivery fee?", [c for c, _ in chunks])
    assert prompt.startswith("Context:\n")
    assert prompt.endswith("Question: What is the delivery fee?")


def test_every_config_builds():
    for config in CONFIGS.values():
        pipeline = QAPipeline(config, MockLLM())
        assert pipeline.ask("delivery fee").answer
