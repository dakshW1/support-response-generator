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
TOY = [["the", "order", "shipped"], ["the", "order", "arrived"], ["the", "refund", "arrived"]]


def test_mle_counts_turn_into_probabilities_without_smoothing():
    model = core.NGramModel(n=2, k=0).fit(TOY)
    assert model.prob("order", ["the"]) == pytest.approx(2 / 3)
    assert model.prob("refund", ["the"]) == pytest.approx(1 / 3)
    assert model.prob("shipped", ["the"]) == 0


def test_add_k_gives_unseen_words_some_probability():
    model = core.NGramModel(n=2, k=1).fit(TOY)
    # V = {the, order, shipped, arrived, refund, </s>, <unk>} = 7 ; C(the) = 3
    assert model.prob("shipped", ["the"]) == pytest.approx((0 + 1) / (3 + 7))
    assert model.prob("never-seen-word", ["the"]) > 0


@pytest.mark.parametrize("smoothing", ["add_k", "interpolated"])
def test_distribution_sums_to_one(smoothing):
    model = core.NGramModel(n=3, k=0.1, smoothing=smoothing).fit(TOY)
    for history in ([], ["the"], ["the", "order"], ["unseen", "words"]):
        total = sum(model.distribution(history).values()) + model.prob(core.UNK, history)
        assert total == pytest.approx(1.0)


def test_smoothing_lowers_perplexity_on_held_out_replies(corpus):
    results = {(r["n"], r["k"], r["smoothing"]): r["perplexity"] for r in core.evaluate(corpus)}
    assert all(math.isfinite(p) and p > 1 for p in results.values())
    assert results[(3, 0.01, "add_k")] < results[(3, 1.0, "add_k")] < results[(1, 1.0, "add_k")]


# Milestone 3: Create a text generation function to suggest response continuations.
def test_suggest_next_ranks_the_common_continuation_first(corpus):
    model = core.train(path=corpus)
    words = [w for w, _ in core.suggest_next(model, "thank you for", top_k=3)]
    assert words[0] == "reaching"
    assert core.suggest_next(model, "you can return the", top_k=1)[0][0] == "item"


def test_greedy_generation_is_repeatable_and_does_not_loop(corpus):
    model = core.train(path=corpus)
    draft = core.generate(model, "i am sorry", max_words=60)
    assert draft.startswith("I am sorry")
    assert draft == core.generate(model, "i am sorry", max_words=60)
    toks = core.tokenize(draft)
    fourgrams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    assert len(fourgrams) == len(set(fourgrams))


def test_sampling_is_seeded_and_category_changes_the_reply(corpus):
    model = core.train(path=corpus)
    assert core.generate(model, "your", sample=True, seed=7) == core.generate(model, "your", sample=True, seed=7)
    refund = core.generate(core.train("refund", path=corpus))
    account = core.generate(core.train("account", path=corpus))
    assert refund != account


# Milestone 4: Build a command-line interface for agents to get suggestions.
from app.cli import interactive, main  # noqa: E402


def test_cli_suggest_and_complete(capsys):
    main(["suggest", "thank you for", "--top", "3"])
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("1. reaching")
    main(["complete", "your refund has been", "-c", "refund"])
    assert capsys.readouterr().out.lower().startswith("your refund has been")


def test_cli_evaluate_prints_a_table(capsys):
    main(["evaluate"])
    out = capsys.readouterr().out
    assert "perplexity" in out and "interpolated" in out


def test_interactive_lets_the_agent_pick_suggestions(corpus):
    model = core.train(path=corpus)
    answers = iter(["thank", "you", "1", ""])  # type two words, pick suggestion 1, finish
    final = interactive(model, inp=lambda _: next(answers), out=lambda *_: None)
    assert final == "Thank you for"
