import pytest

from qa.chunking import _word_windows, fixed_size_chunks, section_chunks

DOCS = {
    "sample.md": "# Title\n\nIntro text here.\n\n## Fees\n\nThe fee is $4.99 per order.\n\n## Hours\n\nOpen 8 to 9.",
}


def test_word_windows_with_overlap():
    words = [str(i) for i in range(10)]
    windows = _word_windows(words, size=4, overlap=2)
    assert windows[0] == ["0", "1", "2", "3"]
    assert windows[1] == ["2", "3", "4", "5"]  # overlaps by 2 words
    assert windows[-1][-1] == "9"  # nothing is lost at the end


def test_word_windows_rejects_bad_overlap():
    with pytest.raises(ValueError):
        _word_windows(["a", "b"], size=2, overlap=2)


def test_fixed_size_chunks_cover_all_words():
    chunks = fixed_size_chunks(DOCS, size=5)
    joined = " ".join(chunk.text for chunk in chunks)
    assert joined.split() == DOCS["sample.md"].split()


def test_section_chunks_label_each_chunk_with_its_section():
    chunks = section_chunks(DOCS, size=50, overlap=5)
    sections = {chunk.section for chunk in chunks}
    assert {"Fees", "Hours"} <= sections
    fees = next(chunk for chunk in chunks if chunk.section == "Fees")
    assert fees.text.startswith("[Title > Fees]")
    assert "$4.99" in fees.text
