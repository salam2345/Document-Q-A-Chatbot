from qa.scoring import is_refusal, keyword_found, score_answer


def test_numbers_with_commas_match():
    assert keyword_found("Free delivery over KES 3,000.", "3000")


def test_short_keywords_match_whole_words_only():
    assert keyword_found("No, cash is not accepted.", "no")
    assert not keyword_found("I know the answer.", "no")


def test_alternatives_separated_by_pipe():
    assert keyword_found("You can use 1 promo code.", "one|1|single")


def test_answerable_question_needs_all_keywords():
    item = {"answerable": True, "expected_keywords": ["7:00", "10:00"]}
    assert score_answer("Chat runs 7:00 a.m. to 10:00 p.m.", item)
    assert not score_answer("Chat opens at 7:00 a.m.", item)


def test_refusal_is_wrong_for_answerable_question():
    item = {"answerable": True, "expected_keywords": ["no"]}
    assert not score_answer("I don't know, no idea.", item)


def test_unanswerable_question_needs_a_refusal():
    item = {"answerable": False, "expected_keywords": []}
    assert score_answer("I don't know based on the Kikapu help documents.", item)
    assert not score_answer("The CEO is Jane Doe.", item)
    assert is_refusal("I don’t know.")  # curly apostrophe
