import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

from discriminate import generate_discriminating_question
from utils import extract_action_content

logger = logging.getLogger(__name__)

# ``ThetaTurnSetup`` appends this many discriminating questions to the sampled
# pool, so it asks the sampler for that many fewer. Keep the subtraction and the
# ``n=`` argument equal.
#
# Equal values do NOT make the pool size constant. The subtraction always
# applies, but the append is guarded (see ``ThetaTurnSetup.prepare``) and the
# result is de-duplicated, so the pool holds 14 questions on turn 0, on the last
# turn, and whenever the posterior holds one concept, and 15 otherwise.
_DISCRIMINATING_QUESTIONS = 1


# Unwraps a malformed action line such as ``ASK[Is it a mammal?]`` when the
# strict ``ASK["..."]`` pattern in ``utils.extract_action_content`` does not
# match.
_ACTION_WRAPPER = re.compile(r'^(?:ASK|GUESS)\s*\[\s*"?(.*?)"?\s*\]?$', re.DOTALL)


def _react_question(react_output: str) -> str:
    """Return the question inside ``Action: ASK["..."]`` of a ReAct reply.

    A ReAct reply opens with a ``Thought:`` line that usually names candidate
    entities. Sending the whole reply to the answerer asks a malformed
    question and, because ``utils.text_mentions_item`` scans the question text for
    the target, lets the guesser's own reasoning trigger the "Bingo!" reply and
    score a correct guess at turn 1. Only the action content may leave here.

    A reply that breaks the format falls back to the text after the last
    ``Action:`` marker, then to the last non-empty line, so the caller still
    receives a question rather than the reasoning trace.
    """
    _, content = extract_action_content(react_output)
    if content:
        return content
    tail = react_output.rsplit("Action:", 1)[-1]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return react_output.strip()
    match = _ACTION_WRAPPER.match(lines[-1])
    return match.group(1).strip() if match else lines[-1]


@dataclass
class TurnSetupData:
    generated_question: Optional[str] = None
    # The text actually put to the answerer, when it differs from
    # ``generated_question``. ReAct keeps the Thought in the guesser transcript
    # but must ask only the action; every other acquisition leaves this None.
    answerer_question: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    followup_questions: List[str] = field(default_factory=list)
    guess: str = "N/A"
    pre_guess_correct: bool = False


class TurnSetup(Protocol):
    def prepare(
        self,
        *,
        turn_index: int,
        num_turns: int,
        user_mode: bool,
        current_concepts: Optional[Sequence[str]] = None,
    ) -> TurnSetupData: ...


class NaiveTurnSetup:
    def __init__(
        self, agent: Any, question_sampler: Any, acquisition: str, num_questions: int
    ):
        self.agent = agent
        self.question_sampler = question_sampler
        self.acquisition = acquisition
        self.num_questions = num_questions

    def prepare(
        self,
        *,
        turn_index: int,
        num_turns: int,
        user_mode: bool,
        current_concepts: Optional[Sequence[str]] = None,
    ) -> TurnSetupData:
        is_auto_mode = not user_mode
        if not is_auto_mode:
            return TurnSetupData()

        answerer_question = None
        if self.acquisition == "Implicit_Max_no_beliefs":
            followup_questions = self.question_sampler.generate_followup_questions(
                self.agent.guesser_messages,
                num_questions=self.num_questions,
                final=(turn_index == num_turns - 1),
                concepts=[],
                dataset=self.agent.task,
                item=self.agent.item,
                log_correct_guess_prob=self.agent.log_correct_guess_prob,
            )
            generated_question = self.agent.select_question_no_beliefs(
                followup_questions
            )
        elif self.acquisition == "React":
            react_output = self.agent.guesser_chat(self.agent.guesser_messages)[
                "content"
            ]
            # only the action is asked.
            generated_question = react_output
            answerer_question = _react_question(react_output)
        else:
            generated_question = self.agent.guesser_chat()["content"]

        if turn_index > 0 and self.agent.first_correct_guess is None:
            guess_request = {
                "role": "user",
                "content": "Based on what you know so far, can you guess what entity I'm thinking of?",
            }
            temp_messages = self.agent.guesser_messages.copy()
            temp_messages.append(guess_request)
            guess = self.agent.guesser_chat(temp_messages)["content"]
            if self.agent._is_correct_guess(guess):
                self.agent._record_first_correct_guess(turn_index + 1)

        return TurnSetupData(
            generated_question=generated_question,
            answerer_question=answerer_question,
        )


class _ConceptBackboneTurnSetup:
    def __init__(
        self,
        agent: Any,
        question_sampler: Any,
        concept_sampler: Any,
        *,
        filter_value: Optional[bool] = None,
    ):
        self.agent = agent
        self.question_sampler = question_sampler
        self.concept_sampler = concept_sampler
        self.filter_value = filter_value
        self.valid_concepts: Optional[List[str]] = None
        self.last_guess = "N/A"

    def _sample_concepts(self, turn_index: int) -> tuple[list[str], float]:
        concept_start_time = time.time()
        concepts: List[str] = []
        for _ in range(3):
            kwargs = {
                "stage": turn_index + 1,
                "valid_concepts": self.valid_concepts,
                "task": self.agent.task,
            }
            if self.filter_value is not None:
                kwargs["filter"] = self.filter_value

            concepts = self.concept_sampler.generate_diverse_samples_from_chat(
                self.agent.guesser_messages,
                **kwargs,
            )
            self.valid_concepts = list(concepts)
            if turn_index == 0:
                break
        concept_response_time = time.time() - concept_start_time
        return list(concepts), concept_response_time

    def _log_generation_metrics(
        self,
        *,
        turn_index: int,
        concept_response_time: float,
        question_response_time: float,
        concepts: Sequence[str],
    ) -> None:
        self.agent._log_event(
            {
                "concept_generation_time": concept_response_time,
                "question_generation_time": question_response_time,
                "custom_step": turn_index + 1,
            },
            step=turn_index + 1,
        )
        self.agent._log_event(
            {
                f"eval/concepts_{self.agent.item}": len(concepts),
                "custom_step": turn_index + 1,
            }
        )

    def _run_pre_guess(self, turn_index: int, concepts: Sequence[str]) -> bool:
        pre_guess_correct = False
        if turn_index > 0 and self.agent.first_correct_guess is None:
            guess, answer_prob = self.question_sampler.generate_followup_questions(
                self.agent.guesser_messages,
                num_questions=1,
                final=True,
                concepts=concepts,
                dataset=self.agent.task,
                item=self.agent.item,
                return_answer_probs=True,
                log_correct_guess_prob=self.agent.log_correct_guess_prob,
            )
            self.last_guess = guess[0]
            if answer_prob is not None:
                overall_answer_prob = np.logaddexp.reduce(answer_prob, axis=0)
                logger.info(
                    "[%s] turn %d overall answer logprob: %s",
                    self.agent.item,
                    turn_index,
                    overall_answer_prob,
                )
                self.agent.turn_by_turn_theta_concept_probs.append(-overall_answer_prob)
                self.agent._log_event(
                    {
                        f"correct_guess_logprob/{self.agent.item}": float(
                            overall_answer_prob
                        )
                    }
                )

            if (
                self.agent._is_correct_guess(self.last_guess)
                and self.agent.first_correct_guess is None
            ):
                self.agent._record_first_correct_guess(turn_index + 1)
                pre_guess_correct = True
        return pre_guess_correct


class EigTurnSetup(_ConceptBackboneTurnSetup):
    def __init__(
        self,
        agent: Any,
        question_sampler: Any,
        concept_sampler: Any,
        *,
        num_questions: int,
        filter_value: bool,
    ):
        super().__init__(
            agent, question_sampler, concept_sampler, filter_value=filter_value
        )
        self.num_questions = num_questions

    def prepare(
        self,
        *,
        turn_index: int,
        num_turns: int,
        user_mode: bool,
        current_concepts: Optional[Sequence[str]] = None,
    ) -> TurnSetupData:
        is_auto_mode = not user_mode
        if not is_auto_mode:
            return TurnSetupData(guess=self.last_guess)

        concepts, concept_response_time = self._sample_concepts(turn_index)
        question_start_time = time.time()
        followup_questions = self.question_sampler.generate_followup_questions(
            self.agent.guesser_messages,
            num_questions=self.num_questions,
            final=(turn_index == num_turns - 1),
            concepts=concepts,
            dataset=self.agent.task,
            item=self.agent.item,
            log_correct_guess_prob=self.agent.log_correct_guess_prob,
        )
        question_response_time = time.time() - question_start_time

        pre_guess_correct = self._run_pre_guess(turn_index, concepts)

        self._log_generation_metrics(
            turn_index=turn_index,
            concept_response_time=concept_response_time,
            question_response_time=question_response_time,
            concepts=concepts,
        )
        return TurnSetupData(
            concepts=concepts,
            followup_questions=list(followup_questions),
            guess=self.last_guess,
            pre_guess_correct=pre_guess_correct,
        )


class ThetaTurnSetup(_ConceptBackboneTurnSetup):
    def __init__(self, agent: Any, question_sampler: Any, *, num_questions: int):
        self.agent = agent
        self.question_sampler = question_sampler
        self.num_questions = num_questions
        self.last_guess = "N/A"

    def prepare(
        self,
        *,
        turn_index: int,
        num_turns: int,
        user_mode: bool,
        current_concepts: Optional[Sequence[str]] = None,
    ) -> TurnSetupData:
        is_auto_mode = not user_mode
        concepts = list(current_concepts or [])
        if not is_auto_mode:
            return TurnSetupData(concepts=concepts, guess=self.last_guess)

        followup_questions = self.question_sampler.generate_followup_questions(
            self.agent.guesser_messages,
            num_questions=self.num_questions - _DISCRIMINATING_QUESTIONS,
            final=(turn_index == num_turns - 1),
            concepts=concepts,
            dataset=self.agent.task,
            item=self.agent.item,
            log_correct_guess_prob=self.agent.log_correct_guess_prob,
        )
        if turn_index < num_turns - 1 and len(concepts) > 1 and turn_index > 0:
            new_question = generate_discriminating_question(
                concepts,
                self.question_sampler.model.client,
                self.question_sampler.model.model_name,
                temperature=1.0,
                n=_DISCRIMINATING_QUESTIONS,
                task=self.agent.task,
            )
            followup_questions += new_question
            followup_questions = list(dict.fromkeys(followup_questions))

        pre_guess_correct = self._run_pre_guess(turn_index, concepts)

        self.agent._log_event(
            {
                f"eval/concepts_{self.agent.item}": len(concepts),
                "custom_step": turn_index + 1,
            }
        )
        return TurnSetupData(
            concepts=concepts,
            followup_questions=list(followup_questions),
            guess=self.last_guess,
            pre_guess_correct=pre_guess_correct,
        )
