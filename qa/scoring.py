"""Decide whether an answer is correct.

Two simple, repeatable checks:

* Answerable questions: the answer must contain every expected keyword
  (case-insensitive; "3,000" and "3000" count as the same).
* Unanswerable questions: the answer must admit it doesn't know, instead of
  making something up (a "hallucination").
"""

from __future__ import annotations

import re

REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "not in the context",
    "not mentioned",
    "no information",
    "doesn't contain",
    "does not contain",
    "doesn't mention",
    "does not mention",
    "not provided",
    "can't find",
    "cannot find",
)


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes and remove thousands separators in numbers."""
    text = text.lower().replace("’", "'")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def keyword_found(answer: str, keyword: str) -> bool:
    """True if the keyword appears as a whole word or phrase in the answer.

    "no" matches "No, ..." but not "know" or "not". A keyword may list
    alternatives separated by "|", e.g. "one|1|single".
    """
    answer = normalize(answer)
    for option in normalize(keyword).split("|"):
        option = option.strip()
        if not option:
            continue
        start = r"\b" if option[0].isalnum() else ""
        end = r"\b" if option[-1].isalnum() else ""
        if re.search(start + re.escape(option) + end, answer):
            return True
    return False


def contains_all(answer: str, keywords: list[str]) -> bool:
    return all(keyword_found(answer, keyword) for keyword in keywords)


def is_refusal(answer: str) -> bool:
    answer = normalize(answer)
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def score_answer(answer: str, item: dict) -> bool:
    """True if the answer is correct for one evaluation item."""
    if item.get("answerable", True):
        return contains_all(answer, item["expected_keywords"]) and not is_refusal(answer)
    return is_refusal(answer)
