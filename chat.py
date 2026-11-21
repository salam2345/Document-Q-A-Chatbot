"""Ask the chatbot questions from the terminal.

    python chat.py                      # uses the improved config
    python chat.py --config baseline
    python chat.py --provider mock      # no API key needed
"""

from __future__ import annotations

import argparse

from qa.llm import make_llm
from qa.pipeline import CONFIGS, QAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Kikapu help documents.")
    parser.add_argument("--config", choices=list(CONFIGS), default="improved")
    parser.add_argument("--provider", choices=["llm", "mock"], default="llm")
    parser.add_argument("--show-sources", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    pipeline = QAPipeline(CONFIGS[args.config], make_llm(args.provider))
    print(f"Kikapu help assistant ({args.config} config). Type a question, or 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = pipeline.ask(question)
        print(f"Bot: {result.answer}")
        sources = sorted({chunk.source for chunk in result.sources})
        print(f"     (sources: {', '.join(sources)})")
        if args.show_sources:
            for chunk in result.sources:
                print(f"     - {chunk.text[:150]}...")
        print()


if __name__ == "__main__":
    main()
