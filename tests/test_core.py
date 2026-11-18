"""Tests: one block per milestone in README.md."""
import csv
import math

import pytest

from app import core


@pytest.fixture
def corpus(tmp_path):
    """A freshly prepared clean corpus in a temp folder (never touches the real one)."""
    out = tmp_path / "support_corpus.tsv"
    core.prepare_corpus(core.RAW_CORPUS, out)
    return out


# Milestone 1: Prepare an anonymized corpus of customer support interactions.
def test_anonymize_masks_personal_details():
    text = "Mail priya.k@example.com or call +91 98450 12345 about order #A48213 and 77120934."
    masked = core.anonymize(text)
    assert "priya" not in masked and "98450" not in masked and "A48213" not in masked
    assert "<email>" in masked and "<phone>" in masked and masked.count("<order_id>") == 2


def test_tokenize_lowercases_splits_punctuation_and_keeps_placeholders():
    assert core.tokenize("Hi there,  your order <order_id> SHIPPED!") == \
        ["hi", "there", ",", "your", "order", "<order_id>", "shipped", "!"]
    assert core.detokenize(["thank", "you", "!", "your", "order", "."]) == "Thank you! Your order."


def test_prepared_corpus_has_no_personal_data(corpus):
    with open(corpus, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) >= 50
    text = " ".join(r["customer"] + " " + r["agent"] for r in rows)
    assert "@" not in text
    assert not any(tok.isdigit() and len(tok) >= 6 for tok in text.split())
    assert text == text.lower()


# Milestone 2: Implement the n-gram model with smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Create a text generation function to suggest response continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a command-line interface for agents to get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

