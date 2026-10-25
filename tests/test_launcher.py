"""eig_game_play: flag resolution, dispatch, ports, and result bookkeeping."""

from types import SimpleNamespace as NS

import pytest
from absl import app

import eig_game_play as egp
from Q20 import Q20
from concept_sampler import ConceptSampler
from question_sampler import QuestionSampler


def test_answerer_defaults_to_the_guesser(run_config):
    cfg = run_config("--model_name=org/a")
    assert cfg.answerer_model_name == "org/a"
    assert cfg.acquisitions == ("EIG",)


def test_removed_flags_are_gone_and_live_ones_remain():
    names = set(egp.FLAGS)
    for gone in (
        "consistency_threshold",
        "declarative_temperature",
        "declarative_top_p",
        "declarative_max_tokens",
        "declarative_repetition_penalty",
    ):
        assert gone not in names
    assert {"use_logprob_filter", "tokens_per_sample", "answerer_port"} <= names


def test_dispatch_builds_kwargs_for_every_acquisition(run_config, fake_model):
    cfg = run_config("--model_name=gpt-4.1")
    model = fake_model("gpt-4.1")
    qs = QuestionSampler(model=model, config=cfg.question_config)
    cs = ConceptSampler(model=model, config=cfg.concept_config)
    for acquisition in egp.VALID_ACQUISITIONS:
        method_name, kwargs = egp._build_dispatch(acquisition, qs, cs, cfg)
        assert hasattr(Q20, method_name), acquisition
        assert "data" not in kwargs, acquisition
        assert kwargs["num_turns"] == cfg.num_turns


def test_dispatch_rejects_an_unknown_acquisition(run_config, fake_model):
    cfg = run_config("--model_name=gpt-4.1")
    with pytest.raises(ValueError, match="Unknown acquisition"):
        egp._build_dispatch("Nope", None, None, cfg)


# ---- ports ----------------------------------------------------------------


def test_answerer_port_defaults_to_port_plus_one(run_config):
    cfg = run_config("--model_name=org/a", "--answerer_model_name=org/b", "--port=8100")
    assert cfg.answerer_port == 8101


def test_answerer_port_flag_is_honoured(run_config):
    cfg = run_config(
        "--model_name=org/a", "--answerer_model_name=org/b", "--port=8100", "--answerer_port=9000"
    )
    assert cfg.answerer_port == 9000


def _port_rejection(cfg):
    try:
        egp._reject_port_collision(cfg)
    except app.UsageError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(
    "args",
    [
        ("--model_name=org/a", "--answerer_model_name=org/b"),
        # Same model: one server, one port.
        ("--model_name=org/a", "--answerer_model_name=org/a", "--answerer_port=8000"),
        # A hosted model on either side uses no port.
        ("--model_name=org/a", "--answerer_model_name=gpt-4.1", "--answerer_port=8000"),
        ("--model_name=gpt-4.1", "--answerer_model_name=org/b", "--answerer_port=8000"),
    ],
)
def test_port_collision_accepts_runs_with_at_most_one_server_per_port(args, run_config):
    assert _port_rejection(run_config(*args)) is None


def test_port_collision_rejects_two_local_models_on_one_port(run_config):
    cfg = run_config(
        "--model_name=org/a", "--answerer_model_name=org/b", "--port=8000", "--answerer_port=8000"
    )
    message = _port_rejection(cfg)
    assert message is not None
    assert "both 8000" in message
    assert "org/a" in message and "org/b" in message


def test_setup_models_gives_distinct_local_models_distinct_ports(run_config, monkeypatch):
    made = []
    monkeypatch.setattr(
        egp,
        "ModelServer",
        lambda model_name, port, num_gpus: made.append((model_name, port))
        or NS(model_name=model_name, port=port),
    )
    guesser, answerer = egp._setup_models(
        run_config("--model_name=org/a", "--answerer_model_name=org/b", "--port=8000")
    )
    assert made == [("org/a", 8000), ("org/b", 8001)]
    assert guesser is not answerer


def test_setup_models_shares_one_server_for_the_same_model(run_config, monkeypatch):
    made = []
    monkeypatch.setattr(
        egp,
        "ModelServer",
        lambda model_name, port, num_gpus: made.append((model_name, port))
        or NS(model_name=model_name, port=port),
    )
    guesser, answerer = egp._setup_models(run_config("--model_name=org/a", "--port=8000"))
    assert made == [("org/a", 8000)]
    assert guesser is answerer


# ---- result bookkeeping ---------------------------------------------------


def test_cumulative_correct_by_turn_counts_first_guesses_at_or_before_each_turn():
    assert egp._cumulative_correct_by_turn([1, 3, 3], num_turns=4) == {1: 1, 2: 1, 3: 3, 4: 3}
    assert egp._cumulative_correct_by_turn([], num_turns=2) == {1: 0, 2: 0}


def test_completed_first_guesses_skips_agents_that_never_guessed():
    agents = {
        "a": NS(first_correct_guess=2),
        "b": NS(first_correct_guess=None),
        "c": NS(),
    }
    assert egp._completed_first_guesses(agents) == [2]


def test_artifact_paths_tag_the_logprob_filter_only_when_on(run_config):
    plain = egp.ArtifactPaths(run_config("--model_name=gpt-4.1", "--dataset=animals"), "ts")
    assert plain.plot().name == "success_rates_gpt-4.1_gpt-4.1_animals.pdf"
    filtered = egp.ArtifactPaths(
        run_config("--model_name=gpt-4.1", "--dataset=animals", "--use_logprob_filter"), "ts"
    )
    assert filtered.plot().name == "success_rates_gpt-4.1_gpt-4.1_animals_lpfilter0.3.pdf"
    assert filtered.data("EIG").name == "data_gpt-4.1_gpt-4.1_animals_EIG__lpfilter0.3_ts.json"
