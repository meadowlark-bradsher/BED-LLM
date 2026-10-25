"""Per-turn setup and the pieces of Q20 that run without a model."""

import inspect
from types import SimpleNamespace as NS

import pytest

from Q20 import Q20
from q20_turn_setup import EigTurnSetup, NaiveTurnSetup


def _agent(messages, **overrides):
    base = dict(
        guesser_messages=messages,
        task="animals",
        item="Lion",
        first_correct_guess=None,
        log_correct_guess_prob=False,
        _log_event=lambda *a, **k: None,
        _is_correct_guess=lambda guess: False,
        _record_first_correct_guess=lambda turn: None,
    )
    base.update(overrides)
    return NS(**base)


def test_eig_turn_setup_consumes_the_sampler_list_directly(chat_messages):
    sampler = NS(generate_diverse_samples_from_chat=lambda messages, **kw: ["Lion", "Tiger"])
    setup = EigTurnSetup(
        _agent(chat_messages), question_sampler=None, concept_sampler=sampler,
        num_questions=3, filter_value=False,
    )
    concepts, elapsed = setup._sample_concepts(turn_index=0)
    assert concepts == ["Lion", "Tiger"]
    assert setup.valid_concepts == ["Lion", "Tiger"]
    assert elapsed >= 0
    assert not hasattr(setup, "declarative_messages")


def test_react_setup_asks_only_the_action_but_records_the_whole_reply(chat_messages):
    reply = 'Thought: it is probably a lion\nAction: ASK["Is it a cat?"]'
    agent = _agent(chat_messages, guesser_chat=lambda messages=None: {"role": "assistant", "content": reply})
    setup = NaiveTurnSetup(agent, question_sampler=None, acquisition="React", num_questions=1)
    data = setup.prepare(turn_index=0, num_turns=20, user_mode=False)
    assert data.generated_question == reply
    assert data.answerer_question == "Is it a cat?"
    assert "lion" not in data.answerer_question.lower()


def test_user_mode_prepares_nothing(chat_messages):
    setup = NaiveTurnSetup(_agent(chat_messages), None, "Prompt_Only", 1)
    data = setup.prepare(turn_index=0, num_turns=20, user_mode=True)
    assert data.generated_question is None


# ---- Q20 without a model ----------------------------------------------------


def test_q20_constructs_without_removed_arguments(fake_model):
    game = Q20(guesser_model=fake_model("gpt-4.1"), answerer_model=fake_model("gpt-4.1"),
               item="Lion", task="animals")
    assert not hasattr(game, "history")
    assert "data" not in inspect.signature(game.game_play_with_eig).parameters
    assert "data" not in inspect.signature(game.game_play_theta_eig).parameters
    game.reset()
    assert game.guesser_messages[0]["role"] == "user"
    assert "animal" in game.guesser_messages[0]["content"]


def test_q20_question_cleaning_and_bingo_detection():
    assert Q20._clean_question_text("Is the entity you are thinking of alive?") == "Is it alive?"
    assert Q20._clean_question_text("Is the person you're thinking of alive?") == "Is the person alive?"
    assert Q20._is_bingo_response({"content": "Bingo!"})
    assert not Q20._is_bingo_response({"content": "No."})


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Yes.", "yes"),
        ("yes, it is", "yes"),
        ("No.", "no"),
        # Uncertain replies keep the current pool. "Don't know." is the
        # --enable_maybe answer and contains "no"; the old substring check
        # sent it down the No branch.
        ("Don't know.", "keep"),
        ("Unknown", "keep"),
        ("Maybe.", "keep"),
        ("Bingo!", "keep"),
    ],
)
def test_q20_theta_branch_follows_the_leading_token(answer, expected):
    pools = {"keep": ["a", "b"], "yes": ["a"], "no": ["b"]}
    chosen = Q20._concepts_after_answer(answer, pools["keep"], pools["yes"], pools["no"])
    assert chosen is pools[expected]


def test_q20_requires_a_generated_question_in_auto_mode(fake_model):
    game = Q20(guesser_model=fake_model("gpt-4.1"), answerer_model=fake_model("gpt-4.1"), item="Lion")
    with pytest.raises(ValueError, match="Missing generated question at turn 3"):
        game._resolve_guesser_message(2, user_mode=False, generated_question=None)
    msg = game._resolve_guesser_message(0, user_mode=False, generated_question="Is it big?")
    assert msg == {"role": "assistant", "content": "Is it big?"}
