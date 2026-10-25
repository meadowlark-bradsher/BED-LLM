"""The two wandb call sites outside the entry point no-op when no run is active.

``wandb.log`` raises before ``wandb.init``, and ``--debug`` never starts a
run, so without the guard the first metric log killed every game.
"""

import pytest
import wandb

import utils
from q20_dependencies import WandbRunLogger


@pytest.fixture
def wandb_calls(monkeypatch):
    """Record wandb.log calls, raising before init exactly as wandb does."""
    calls = []

    def _log(payload, step=None):
        if wandb.run is None:
            raise RuntimeError("You must call wandb.init() before wandb.log()")
        calls.append((payload, step))

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "log", _log)
    monkeypatch.setattr(wandb, "Histogram", lambda x: ("hist", list(x)))
    return calls


class _Agent:
    """q1 splits the two concepts perfectly; q2 is uninformative."""

    def yes_no_probs(self, question, concept, use_history=False):
        if question == "q1":
            return (1.0, 0.0) if concept == "c1" else (0.0, 1.0)
        return (0.5, 0.5)


def test_run_logger_drops_events_before_init(wandb_calls):
    WandbRunLogger(wandb).log_event({"a": 1}, step=3)
    assert wandb_calls == []


def test_run_logger_forwards_events_after_init(wandb_calls, monkeypatch):
    monkeypatch.setattr(wandb, "run", object())
    WandbRunLogger(wandb).log_event({"a": 1}, step=3)
    WandbRunLogger(wandb).log_event({"b": 2})
    assert wandb_calls == [({"a": 1}, 3), ({"b": 2}, None)]


def test_question_eig_skips_histogram_before_init_and_still_chooses(wandb_calls):
    question, eig = utils.question_eig(
        ["c1", "c2"], ["q1", "q2"], _Agent(), item="x", step=1
    )
    assert question == "q1"
    assert eig == pytest.approx(1.0)
    assert wandb_calls == []


def test_question_eig_logs_histogram_after_init(wandb_calls, monkeypatch):
    monkeypatch.setattr(wandb, "run", object())
    utils.question_eig(["c1", "c2"], ["q1", "q2"], _Agent(), item="x", step=4)
    assert len(wandb_calls) == 1
    payload, step = wandb_calls[0]
    assert payload["custom_step"] == 4
    assert "average_entropy_hist/x" in payload
    assert step is None
