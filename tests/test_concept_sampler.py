"""ConceptSampler: request assembly, parsing, the pool contract, and the filter."""

import math
from types import SimpleNamespace as NS

import pytest

import concept_sampler as cs_mod
from concept_sampler import ConceptSampler, ConceptSamplerConfig


class ChatStub:
    """A chat.completions stand-in that returns ``text`` for every call."""

    def __init__(self, text):
        self.text = text
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        n = kwargs.get("n", 1)
        return NS(choices=[NS(message=NS(content=self.text))] * n)


@pytest.fixture
def sampler():
    chat = ChatStub("Lion\nTiger\nBear")
    model = NS(model_name="org/model", client=NS(chat=NS(completions=chat)))
    sampler = ConceptSampler(model=model, config=ConceptSamplerConfig(n_completions=1))
    sampler.chat = chat
    return sampler


def test_latest_qa_pair_indexes_by_stage(sampler, chat_messages):
    assert sampler.latest_qa_pair(chat_messages, stage=2) == ("Is it a living thing?", "Yes.")
    assert sampler.latest_qa_pair(chat_messages, stage=3) == ("Is it a plant?", "No.")
    with pytest.raises(ValueError, match="stage must be 2 or more"):
        sampler.latest_qa_pair(chat_messages, stage=1)
    with pytest.raises(ValueError, match="needs 3 completed turns"):
        sampler.latest_qa_pair(chat_messages, stage=4)


def test_reverse_pairs_keeps_the_head_and_reverses_turn_order(sampler, chat_messages):
    out = sampler.reverse_pairs(chat_messages)
    assert out[0] == chat_messages[0]
    assert [m["content"] for m in out[1:]] == [
        "Is it a plant?", "No.", "Is it a living thing?", "Yes."
    ]


def test_build_sample_request_stage_one_drops_the_head(sampler, chat_messages):
    request = sampler.build_sample_request(
        messages=chat_messages, task="animals", num_samples=5, stage=1, valid_samples=[]
    )
    assert request[0]["content"] == "Is it a plant?"
    assert "5 diverse" in request[-1]["content"]


def test_build_sample_request_later_stages_list_kept_examples(sampler, chat_messages):
    request = sampler.build_sample_request(
        messages=chat_messages, task="animals", num_samples=5, stage=3, valid_samples=["Lion"]
    )
    assert request[0] == chat_messages[0]
    assert "already been generated: ['Lion']" in request[-1]["content"]


def test_parse_concepts_deduplicates_and_drops_noise(sampler):
    out = sampler._parse_concepts("<think>x</think>\nHere:\n1. Lion\nNote: etc\n- Lion\nTiger")
    assert sorted(out) == ["Lion", "Tiger"]


def test_generate_returns_a_plain_list_at_stage_one(sampler, chat_messages):
    out = sampler.generate_diverse_samples_from_chat(chat_messages, stage=1, task="animals")
    assert isinstance(out, list)
    assert sorted(out) == ["Bear", "Lion", "Tiger"]
    # Stage 1 oversamples by the configured factor.
    assert sampler.chat.requests[0]["max_tokens"] == 3 * 15 * 15


def test_generate_merges_kept_concepts_at_later_stages(sampler, chat_messages):
    out = sampler.generate_diverse_samples_from_chat(
        chat_messages, stage=3, task="animals", valid_concepts=["Wolf"], filter=False
    )
    assert isinstance(out, list)
    assert sorted(out) == ["Bear", "Lion", "Tiger"]


def test_generate_rejects_a_missing_stage_or_unknown_task(sampler, chat_messages):
    with pytest.raises(ValueError, match="stage is required"):
        sampler.generate_diverse_samples_from_chat(chat_messages, task="animals")
    with pytest.raises(ValueError, match="Unsupported task"):
        sampler.generate_diverse_samples_from_chat(chat_messages, stage=1, task="planets")


# ---- logprob filter -------------------------------------------------------


@pytest.fixture
def lp_sampler(fake_model):
    config = ConceptSamplerConfig(use_logprob_filter=True, logprob_reject_threshold=0.3)
    return ConceptSampler(model=fake_model("org/model"), config=config)


def test_logprob_filter_rejects_above_threshold_and_keeps_a_tie(lp_sampler):
    keep = lp_sampler._survives_logprob_filter
    yes_heavy = {"Yes": math.log(0.7), "No": math.log(0.3)}
    assert keep(yes_heavy, expected=True, concept="c", question="q") is True
    assert keep(yes_heavy, expected=False, concept="c", question="q") is False
    tight = {"Yes": math.log(0.65), "No": math.log(0.35)}
    assert keep(tight, expected=True, concept="c", question="q") is False


def test_logprob_filter_switches_off_only_on_enough_unusable_replies(lp_sampler):
    for _ in range(cs_mod._LOGPROB_FILTER_MIN_SAMPLES - 1):
        lp_sampler._record_logprob_reply(usable=False)
    assert lp_sampler._logprob_filter_enabled()
    lp_sampler._record_logprob_reply(usable=False)
    assert not lp_sampler._logprob_filter_enabled()


def test_logprob_filter_stays_on_when_most_replies_are_usable(lp_sampler):
    for _ in range(cs_mod._LOGPROB_FILTER_MIN_SAMPLES):
        lp_sampler._record_logprob_reply(usable=False)
        lp_sampler._record_logprob_reply(usable=True)
    assert lp_sampler._logprob_filter_enabled()


def test_logprob_filter_is_off_from_the_start_for_reasoning_models(fake_model):
    config = ConceptSamplerConfig(use_logprob_filter=True)
    sampler = ConceptSampler(model=fake_model("o3-mini"), config=config)
    assert not sampler._logprob_filter_enabled()
