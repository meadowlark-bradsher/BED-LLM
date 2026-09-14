from typing import Any, Callable, Final, Optional, Protocol, Sequence, Tuple


# Single source of truth for acquisition-strategy identifiers.
# The CLI enum, the multi-acquisition validator, and the per-strategy
# dispatch in `eig_game_play._build_dispatch` all draw from this tuple.
# Adding a new strategy: append the identifier here and add its dispatch
# branch in `_build_dispatch`.
VALID_ACQUISITIONS: Final[tuple[str, ...]] = (
    "EIG",
    "Entropy",
    "Prompt_Only",
    "Implicit_Max",
    "Data_estimation_rank_aligned",
    "Implicit_Max_no_beliefs",
    "No_filter",
    "Split_uniform",
    "Data_estimation_union_mixture",
    "Split_uniform_original",
    "Split_uniform_base",
    "Split_logprobs_original",
    "Split_logprobs",
    "Split_logprobs_renormalized",
    "React",
    "Sample_EIG",
)

# Sentinel EIG score for choosers that do not compute an information gain
# (e.g. the LLM-selection path). NaN is the conventional "not measured"
# marker so it is never mistaken for a genuine low EIG when scores are
# logged or aggregated.
NO_SCORE: Final[float] = float("nan")


class QuestionChooser(Protocol):
    def choose(
        self,
        concepts: Sequence[str],
        followup_questions: Sequence[str],
        *,
        agent: Any,
        item: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[str, float]: ...


class PartitionQuestionChooser(Protocol):
    def choose(
        self,
        concepts: Sequence[str],
        followup_questions: Sequence[str],
        *,
        agent: Any,
        item: Optional[str] = None,
        step: Optional[int] = None,
        concept_sampler: Any = None,
    ) -> Tuple[str, float, Sequence[str], Sequence[str]]: ...


class FunctionQuestionChooser:
    def __init__(
        self, chooser_fn: Callable[..., Tuple[str, float]], **fixed_kwargs: Any
    ):
        self._chooser_fn = chooser_fn
        self._fixed_kwargs = dict(fixed_kwargs)

    def choose(
        self,
        concepts: Sequence[str],
        followup_questions: Sequence[str],
        *,
        agent: Any,
        item: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[str, float]:
        return self._chooser_fn(
            concepts,
            followup_questions,
            agent,
            item=item,
            step=step,
            **self._fixed_kwargs,
        )


class LlmQuestionChooser:
    def __init__(self, select_question_fn: Callable[..., str]):
        self._select_question_fn = select_question_fn

    def choose(
        self,
        concepts: Sequence[str],
        followup_questions: Sequence[str],
        *,
        agent: Any,
        item: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[str, float]:
        question = self._select_question_fn(
            followup_questions,
            history=agent.guesser_messages,
            temperature=1.0,
            explain=False,
            concepts=concepts,
        )
        return question, NO_SCORE


class FunctionPartitionQuestionChooser:
    def __init__(
        self, chooser_fn: Callable[..., Tuple[str, float, Sequence[str], Sequence[str]]]
    ):
        self._chooser_fn = chooser_fn

    def choose(
        self,
        concepts: Sequence[str],
        followup_questions: Sequence[str],
        *,
        agent: Any,
        item: Optional[str] = None,
        step: Optional[int] = None,
        concept_sampler: Any = None,
    ) -> Tuple[str, float, Sequence[str], Sequence[str]]:
        return self._chooser_fn(
            concepts,
            followup_questions,
            agent,
            item=item,
            step=step,
            concept_sampler=concept_sampler,
        )
