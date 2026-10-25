"""One predicate decides hosted-vs-local; reasoning models fail fast.

Three sites used to route on a bare ``startswith("gpt")``, which sent gpt-5
into the chat-logprob path and o1/o3/o4 into the local-tokenizer path. Both
reject every request, ten retries deep.
"""

from types import SimpleNamespace as NS

import pytest
from absl import app

import eig_game_play as egp
import serve_model
import theta_eig
import theta_eig_completions
import utils
from Q20 import Q20
from question_sampler import QuestionSampler

HOSTED_CHAT = ["gpt-4.1", "gpt-4o-mini"]
HOSTED_REASONING = ["gpt-5", "o3-mini", "o4-mini", "o1"]
LOCAL = ["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]


@pytest.mark.parametrize(
    "name,hosted,logprobs",
    [(n, True, True) for n in HOSTED_CHAT]
    + [(n, True, False) for n in HOSTED_REASONING]
    + [(n, False, True) for n in LOCAL],
)
def test_predicates_agree_across_every_site(name, hosted, logprobs, fake_model):
    assert serve_model.is_hosted_openai_model(name) is hosted
    assert serve_model.supports_token_logprobs(name) is logprobs
    assert Q20._is_api_model(name) is hosted
    sampler = QuestionSampler(model=fake_model(name))
    assert sampler._supports_completion_logprobs() is (not hosted)


@pytest.fixture
def routed(monkeypatch):
    """Replace both scoring backends with recorders."""
    seen = {}
    monkeypatch.setattr(
        utils,
        "item_probs_from_chat_logprobs",
        lambda model, items, messages=None, task=None: seen.setdefault("chat", True)
        or [0.5, 0.5],
    )
    monkeypatch.setattr(
        utils,
        "item_probs_from_completion_logprobs",
        lambda **kw: seen.setdefault("completions", True) or [0.5, 0.5],
    )
    return seen


def _agent(name, fake_model):
    return NS(task="animals", guesser_model=fake_model(name))


def test_concept_probs_route_hosted_chat_models_to_chat_logprobs(routed, fake_model):
    utils._concept_probs_given_history(
        _agent("gpt-4.1", fake_model), [{"role": "user", "content": "hi"}], ["a", "b"]
    )
    assert routed == {"chat": True}


def test_concept_probs_route_local_models_to_completion_logprobs(routed, fake_model):
    utils._concept_probs_given_history(
        _agent(LOCAL[0], fake_model), [{"role": "user", "content": "hi"}], ["a", "b"]
    )
    assert routed == {"completions": True}


@pytest.mark.parametrize("name", HOSTED_REASONING)
def test_concept_probs_reject_reasoning_models_before_any_call(name, routed, fake_model):
    with pytest.raises(ValueError, match="reasoning model"):
        utils._concept_probs_given_history(_agent(name, fake_model), [], ["a"])
    assert routed == {}


@pytest.mark.parametrize("name", HOSTED_REASONING + LOCAL)
def test_chat_logprob_scorer_rejects_what_it_cannot_serve(name):
    calls = []
    model = NS(
        model_name=name,
        client=NS(chat=NS(completions=NS(create=lambda **k: calls.append(k)))),
    )
    with pytest.raises(ValueError):
        theta_eig.item_probs_from_chat_logprobs(model, ["x"])
    assert calls == []


@pytest.mark.parametrize("name", HOSTED_CHAT + HOSTED_REASONING)
def test_completion_tokenizer_rejects_hosted_models(name):
    theta_eig_completions._tokenizer_for.cache_clear()
    with pytest.raises(ValueError, match="hosted OpenAI model"):
        theta_eig_completions._tokenizer_for(name)


def test_completion_tokenizer_loads_local_models(monkeypatch):
    theta_eig_completions._tokenizer_for.cache_clear()
    monkeypatch.setattr(
        theta_eig_completions.AutoTokenizer, "from_pretrained", lambda m: ("hf", m)
    )
    assert theta_eig_completions._tokenizer_for(LOCAL[0]) == ("hf", LOCAL[0])


# ---- launch-time guard ----------------------------------------------------


def _rejection(cfg):
    try:
        egp._reject_unsupported_logprob_runs(cfg)
    except app.UsageError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(
    "args",
    [
        ("--model_name=gpt-4.1", "--answerer_model_name=gpt-4.1", "--acquisition=EIG"),
        # EIG never reads answerer logprobs, so a reasoning answerer is fine.
        (f"--model_name={LOCAL[0]}", "--answerer_model_name=gpt-5", "--acquisition=EIG"),
        # Text-only acquisitions accept a reasoning guesser.
        ("--model_name=gpt-5", "--answerer_model_name=gpt-4.1",
         "--multi_acquisitions=Prompt_Only,React,Sample_EIG,Implicit_Max,Split_uniform"),
        # Proportion entropy counts text replies.
        ("--model_name=gpt-5", "--answerer_model_name=gpt-4.1",
         "--acquisition=Entropy", "--nouse_logits_entropy"),
    ],
)
def test_launch_guard_accepts_runs_that_need_no_reasoning_logprobs(args, run_config):
    assert _rejection(run_config(*args)) is None


def test_launch_guard_names_the_blocked_acquisitions_only(run_config):
    cfg = run_config(
        "--model_name=gpt-5",
        "--answerer_model_name=gpt-4.1",
        "--multi_acquisitions=EIG,Prompt_Only,Split_logprobs",
    )
    message = _rejection(cfg)
    assert message is not None
    assert "guesser" in message
    assert "EIG, Split_logprobs" in message
    blocked_clause = message.split("needs them for ")[1].split(".")[0]
    assert "Prompt_Only" not in blocked_clause


def test_launch_guard_counts_logits_entropy_as_a_logprob_acquisition(run_config):
    cfg = run_config("--model_name=gpt-5", "--answerer_model_name=gpt-4.1", "--acquisition=Entropy")
    message = _rejection(cfg)
    assert message is not None and "Entropy" in message


def test_launch_guard_checks_the_answerer_for_theta_eig(run_config):
    cfg = run_config(
        "--model_name=gpt-4.1",
        "--answerer_model_name=o3-mini",
        "--acquisition=Data_estimation_rank_aligned",
    )
    message = _rejection(cfg)
    assert message is not None
    assert "answerer" in message
    assert "Data_estimation_rank_aligned" in message
