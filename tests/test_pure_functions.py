"""The math and parsing helpers that take no model and need no network."""

import math
from types import SimpleNamespace as NS

import pytest

from concept_sampler import _clean_concept_line, _qa_pairs
from q20_llm_io import (
    extract_yes_no_probs,
    has_yes_no_mass,
    top_logprob_dict,
    yes_no_mass,
    yes_no_polarity,
)
from q20_turn_setup import _react_question
from question_sampler import _clean_question_line, _split_item_aliases
from utils import (
    _safe_entropy_term,
    average_binary_entropy,
    ensemble_binary_entropy,
    extract_action_content,
    format_conversation,
    parse_llm_line_list,
    text_mentions_item,
)


# ---- yes/no logprob extraction -------------------------------------------


def test_extract_yes_no_probs_renormalises_over_yes_and_no_only():
    p_yes, p_no = extract_yes_no_probs(
        {"Yes": math.log(0.6), "No": math.log(0.2), "Maybe": math.log(0.2)}
    )
    assert p_yes == pytest.approx(0.75)
    assert p_no == pytest.approx(0.25)


def test_extract_yes_no_probs_sums_alternative_spellings():
    p_yes, p_no = extract_yes_no_probs(
        {"Yes": math.log(0.3), "▁Yes": math.log(0.3), "No": math.log(0.4)}
    )
    assert p_yes == pytest.approx(0.6)
    assert p_no == pytest.approx(0.4)


def test_extract_yes_no_probs_floors_a_missing_answer_at_epsilon():
    p_yes, p_no = extract_yes_no_probs({"Yes": math.log(0.5)}, epsilon=1e-6)
    assert p_no == pytest.approx(1e-6 / (0.5 + 1e-6))
    assert p_yes + p_no == pytest.approx(1.0)


def test_extract_yes_no_probs_dead_answers_give_no_information():
    assert extract_yes_no_probs({"Yes": -math.inf, "No": -math.inf}) == (0.5, 0.5)
    assert extract_yes_no_probs({}) == (0.5, 0.5)
    # One dead answer is still decisive.
    assert extract_yes_no_probs({"Yes": -math.inf}) == (0.0, 1.0)


def test_has_yes_no_mass_applies_the_floor_before_renormalising():
    assert has_yes_no_mass({"Yes": math.log(0.04), "No": math.log(0.02)})
    assert not has_yes_no_mass({"Yes": math.log(0.04)})
    assert not has_yes_no_mass({"Yes": -math.inf, "No": "not a number"})
    assert yes_no_mass({"<think>": math.log(0.99)}) == 0.0


def test_top_logprob_dict_flattens_chat_entries():
    entries = [NS(token="Yes", logprob=-0.1), NS(token="No", logprob=-2.0)]
    assert top_logprob_dict(entries) == {"Yes": -0.1, "No": -2.0}


# ---- entropies in bits ----------------------------------------------------


def test_ensemble_entropy_is_entropy_of_the_average():
    assert ensemble_binary_entropy([0.5, 0.5]) == pytest.approx(1.0)
    # Two certain but opposite answers average to 0.5: one full bit.
    assert ensemble_binary_entropy([1.0, 0.0]) == pytest.approx(1.0)
    assert ensemble_binary_entropy([1.0, 1.0]) == 0.0
    assert ensemble_binary_entropy([]) == 0.0


def test_average_entropy_is_mean_of_per_concept_entropies():
    assert average_binary_entropy([(0.5, 0.5), (1.0, 0.0)]) == pytest.approx(0.5)
    assert average_binary_entropy([]) == 0.0


def test_safe_entropy_term_is_zero_and_float_at_the_limits():
    for p in (0.0, 1.0):
        term = _safe_entropy_term(p)
        assert term == 0.0
        assert isinstance(term, float)


def test_eig_identity_rewards_a_question_that_splits_the_concepts():
    results = [(1.0, 0.0), (0.0, 1.0)]
    probs_yes = [p_yes for p_yes, _ in results]
    eig = ensemble_binary_entropy(probs_yes) - average_binary_entropy(results)
    assert eig == pytest.approx(1.0)


# ---- text matching and parsing --------------------------------------------


def test_text_mentions_item_matches_any_alias_literally():
    item = "J.K. Rowling | Joanne Rowling"
    assert text_mentions_item(item, "Is it J.K. Rowling?")
    assert text_mentions_item(item, "is it joanne rowling")
    assert not text_mentions_item(item, "Is it Rowling?")
    # The dots are literal, not regex wildcards.
    assert not text_mentions_item(item, "Is it JxKx Rowling?")


def test_text_mentions_item_respects_word_boundaries():
    assert not text_mentions_item("Oar", "Is it a boar?")
    assert text_mentions_item("Oar", "Is it an oar?")


def test_parse_llm_line_list_drops_think_block_and_preamble():
    text = "<think>reasoning</think>\nHere are some:\n1. Lion\n\n2. Tiger\n"
    out = parse_llm_line_list(
        text,
        boundary_prefixes=("Here",),
        clean_line=lambda line: line.strip().lstrip("0123456789. "),
    )
    assert out == ["Lion", "Tiger"]


def test_parse_llm_line_list_without_boundary_keeps_every_line():
    out = parse_llm_line_list("A\nB", boundary_prefixes=("Here",), clean_line=str.strip)
    assert out == ["A", "B"]


def test_clean_concept_line():
    assert _clean_concept_line("1. Dog (canine)") == "Dog"
    assert _clean_concept_line("- **Blue Whale**") == "Blue Whale"
    assert _clean_concept_line("Assistant: Cat<|END_RESPONSE|>") == "Cat"
    assert _clean_concept_line("Note: these are examples") == ""
    assert _clean_concept_line("   ") == ""


def test_clean_question_line():
    assert _clean_question_line("3. Is it alive?<|END_RESPONSE|>") == "Is it alive?"
    assert _clean_question_line("- Is it big?") == "Is it big?"


def test_split_item_aliases_strips_whitespace():
    assert _split_item_aliases("J.K. Rowling | Joanne Rowling") == [
        "J.K. Rowling",
        "Joanne Rowling",
    ]


def test_qa_pairs_skips_the_head_and_an_unanswered_question():
    messages = [
        {"role": "user", "content": "play"},
        {"role": "assistant", "content": " Q1 "},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "content": "Q2"},
    ]
    assert _qa_pairs(messages) == [("Q1", "Yes.")]


def test_yes_no_polarity_reads_the_leading_token_only():
    assert yes_no_polarity("Yes, it is.") is True
    assert yes_no_polarity(" no.") is False
    assert yes_no_polarity("It depends.") is None
    # Containing the letters is not answering.
    assert yes_no_polarity("Don't know.") is None
    assert yes_no_polarity("Unknown") is None
    assert yes_no_polarity("I am not sure, yes or no") is None


def test_extract_action_content():
    text = 'Thought: hmm\nAction: ASK["Is it a mammal?"]'
    assert extract_action_content(text) == ("ASK", "Is it a mammal?")
    assert extract_action_content("no action here") == (None, None)


def test_react_question_never_returns_the_thought():
    assert _react_question('Thought: a lion\nAction: ASK["Is it a cat?"]') == "Is it a cat?"
    # Unquoted action falls back to the wrapper pattern.
    assert _react_question("Thought: a lion\nAction: ASK[Is it a cat?]") == "Is it a cat?"
    # No action wrapper at all: last non-empty line after the marker.
    assert _react_question("Thought: a lion\nAction:\nIs it a cat?") == "Is it a cat?"
    assert _react_question("Is it a cat?") == "Is it a cat?"


def test_format_conversation():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert format_conversation(messages) == "User: hi\nAssistant: yo"
