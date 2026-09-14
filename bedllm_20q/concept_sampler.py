import functools
import logging
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from q20_llm_io import (
    Q20PromptBuilder,
    extract_yes_no_probs,
    has_yes_no_mass,
    top_logprob_dict,
)
from retry_utils import retry_with_backoff
from serve_model import is_reasoning_model, resolve_model
from utils import (
    format_conversation,
    get_animal_concept_prompt,
    parse_llm_line_list,
    score_ranked_list,
)


logger = logging.getLogger(__name__)


_YES_NO_MAX_TOKENS = 3
_YES_NO_TOP_LOGPROBS = 5
_GREEDY_TEMPERATURE = 0.0

# Cap on consistency-filter fan-out.
_FILTER_MAX_WORKERS = 16

# Top tokens the logprob filter asks for. It matches the EIG call in
# ``q20_llm_io._yes_no_probs_via_chat`` so the filter thresholds the same
# distribution the acquisition integrates.
_FILTER_TOP_LOGPROBS = 10

# The logprob filter reads the answer token only, so one token is enough. A
# larger budget runs a full completion for every concept and every Q/A pair.
_FILTER_LOGPROB_MAX_TOKENS = 1

# Evidence needed before the logprob filter switches itself off. A rejected
# request and a reply with too little yes/no mass both count as unusable.
#
# A count alone is not enough. One ConceptSampler serves every concurrent game,
# so a run of unusable replies says nothing about the backend: it can be the
# order in which threads happen to finish. A single unusable reply is not enough
# either, because a 400 also answers a per-request problem such as an over-long
# prompt. The rule therefore needs both a sample count and a high share. A
# backend that cannot serve logprobs, and a model that always thinks first, both
# sit at a share of 1.0 and switch the mode off after the first
# ``_LOGPROB_FILTER_MIN_SAMPLES`` replies. A model that answers in words most of
# the time stays below the share and keeps the mode for the whole run.
_LOGPROB_FILTER_MIN_SAMPLES = 32
_LOGPROB_FILTER_MAX_UNUSABLE_SHARE = 0.9

# HTTP statuses that mean "this request will never work". A gateway that does not
# implement chat-completions logprobs answers with one of these, so retrying it
# only burns the backoff: the shipped policy sleeps up to 383 seconds per
# concept. Every other status keeps the retry, because a 429 or a 5xx is
# transient.
_UNSUPPORTED_REQUEST_STATUSES: frozenset[int] = frozenset({400, 404, 422})


def _is_unsupported_request(exc: BaseException) -> bool:
    """Return True when the API rejected the request itself, not the server state."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _UNSUPPORTED_REQUEST_STATUSES


def _filter_max_workers(concepts: list[str]) -> int:
    return max(1, min(len(concepts), _FILTER_MAX_WORKERS))


# Stateless; shared so the consistency filter and the answerer cannot drift.
_PROMPT_BUILDER = Q20PromptBuilder()


def _yes_no_polarity(text: str) -> Optional[bool]:
    """Return True for a yes-answer, False for a no-answer, None otherwise.

    Compares only the leading yes/no token, so a verbose reply ("Yes, it is a
    mammal.") is still classified by polarity.
    """
    stripped = text.strip().lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    return None


def _qa_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """Return the completed (question, answer) pairs of a transcript.

    Assumes the ``[head, q1, a1, q2, a2, ...]`` layout used throughout the
    codebase: the first message is not part of a pair, and a trailing question
    with no answer yet is skipped. The contents are stripped, because every
    caller compares or prompts with them as plain text.
    """
    return [
        (messages[i]["content"].strip(), messages[i + 1]["content"].strip())
        for i in range(1, len(messages) - 1, 2)
    ]


_NOISE_PREFIXES: tuple[str, ...] = (
    "Since",
    "Note",
    "(Note",
    "Corrected",
    "These",
    "Please",
    "Sure",
    "Wait",
    "This",
)

_BOUNDARY_PREFIXES: tuple[str, ...] = ("Based", "Here")

# The role and numbering prefix pass. It is separate from the patterns below
# because it runs first and is the only pass that ``.strip()`` follows.
_ROLE_PREFIX_PATTERN: tuple[re.Pattern, str] = (
    re.compile(r"^(Assistant:|User:|\d+\.\s*)"),
    "",
)

# Applied in order after ``_ROLE_PREFIX_PATTERN``, with no strip between them.
_LINE_CLEAN_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\s*\([^)]*\)"), ""),
    (re.compile(r"\*"), ""),
    (re.compile(r"<end_of_turn>"), ""),
    (re.compile(r"^-\s*"), ""),
)


def _clean_concept_line(line: str) -> str:
    """Clean one raw output line into a concept, or ``""`` to drop the line.

    The order of the passes is load-bearing: only ``_ROLE_PREFIX_PATTERN`` is
    followed by ``.strip()``, so the ``_NOISE_PREFIXES`` check sees unstripped
    text.
    """
    pattern, repl = _ROLE_PREFIX_PATTERN
    cleaned = pattern.sub(repl, line).strip()
    for pattern, repl in _LINE_CLEAN_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = cleaned.split("<|END_RESPONSE|>")[0]
    if cleaned.startswith(_NOISE_PREFIXES):
        return ""
    return cleaned.strip()


@dataclass
class ConceptSamplerConfig:
    """Hyperparameters for concept generation and filtering."""

    num_samples: int = 15
    temperature: float = 0.7
    n_completions: int = 3
    stage1_oversample_factor: int = 3
    max_iterations: int = 2
    tokens_per_sample: int = 15
    consistency_threshold: float = 0.8
    multi_completion_stage_cutoff: int = 19
    filter_oversample_factor: int = 3
    chat_retry_max_attempts: int = 10
    chat_retry_initial_delay: float = 1.0
    chat_retry_max_delay: float = 128.0
    chat_retry_backoff_multiplier: float = 2.0
    chat_answer_max_tokens: int = 100
    # Switch the consistency filter from the text yes/no reply to a threshold on
    # the model's confidence that the concept contradicts the observed answer.
    # Off by default. See ``ConceptSampler._survives_logprob_filter``.
    use_logprob_filter: bool = False
    logprob_reject_threshold: float = 0.3


@dataclass(frozen=True)
class _ConceptPromptStrategy:
    initial: Callable[[int], str]
    iterative: Callable[[int], str]


def _celebrities_initial(num_samples: int) -> str:
    return (
        f"Generate a list of {num_samples} diverse, unique, representative examples of people. Some of these people should be more obscure. "
        "Each example should be on a separate line and be a single distinct entity with nothing else added (e.g., 'George Washington', 'Blaise Pascal', 'Luke Skywalker'), and no repetition."
    )


def _celebrities_iterative(num_samples: int) -> str:
    return (
        "You are participating in a game of 20 questions."
        f"Based on the information provided in the questions and answers above, generate a list of **up to** {num_samples} diverse, unique examples of historical or current famous figures. Each person must be consistent with *all* the established clues from the conversation. "
        "Each example should be on a separate line and be a single distinct entity with no extra text added (e.g., 'George Washington', 'Blaise Pascal', 'James Brown'), and no repetition. "
        "Do not include any additional commentary or explanations - **only** provide a famous figure on each line. "
        "Ensure diversity in terms of time periods, geographic locations, and areas of achievement, while strictly adhering to the constraints set by the previous questions and answers. Some of the examples should be more obscure."
    )


def _things_initial(num_samples: int) -> str:
    return (
        f"Generate a list of {num_samples} diverse, unique, representative examples of things. Some of these things should be more obscure. "
        "Each example should be on a separate line and be a single distinct entity with nothing else added (e.g., 'Europa', 'Bagpipe', 'Diadem'), and no repetition."
    )


def _things_iterative(num_samples: int) -> str:
    return f"""You are playing a game of 20 Questions. Using all of the clues established so far:

- Generate up to {num_samples} candidate entities consistent with *all* the clues from the questions and answers in the conversation.
- Each candidate must be a single, self-contained entity (e.g., “Europa”, “Bagpipe”, “Diadem”).
- List each entity on its own line—no numbering, punctuation, or extra text. Include only the entity name on each line and no other text.
- Include a mix of well-known and obscure examples, varying in size, category, and attributes.
- Entities may be objects, living things, places, or other proper nouns, as long as they strictly match every clue provided by the previous questions and answers.
- Do not repeat any entity.

Return only the list of entities."""


_CONCEPT_PROMPTS: dict[str, _ConceptPromptStrategy] = {
    "celebrities": _ConceptPromptStrategy(
        initial=_celebrities_initial,
        iterative=_celebrities_iterative,
    ),
    "animals": _ConceptPromptStrategy(
        initial=lambda n: get_animal_concept_prompt(n, init=True),
        iterative=lambda n: get_animal_concept_prompt(n),
    ),
    "things": _ConceptPromptStrategy(
        initial=_things_initial,
        iterative=_things_iterative,
    ),
}


class _EmptyResponse(Exception):
    """Internal signal: the call succeeded but produced no usable content."""


def chat_with_retries(method=None, *, retry_on_empty: bool = True):
    """Retry a bound method with exponential backoff per ``self.config``.

    Thin adapter over :func:`retry_utils.retry_with_backoff`. Retries on an
    exception and, when ``retry_on_empty`` is true, on a blank return value
    (empty, ``None`` or whitespace-only). Reads the four ``chat_retry_*``
    fields from ``self.config``. Usable bare (``@chat_with_retries``) or
    parameterised (``@chat_with_retries(retry_on_empty=False)``); the
    consistency check passes ``retry_on_empty=False`` so a blank response
    returns at once instead of entering the slow backoff loop.

    On exhaustion it swallows the final exception and returns ``""``. Callers
    therefore cannot tell an API failure from a genuinely empty model response,
    and ``check_single_answer`` reads ``""`` as "keep this concept" - so a dead
    server turns the consistency filter into a no-op instead of stopping the
    run.
    """

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cfg = self.config

            def call_once():
                result = method(self, *args, **kwargs)
                if retry_on_empty and not (result or "").strip():
                    raise _EmptyResponse()
                return result

            def on_error(attempt: int, max_attempts: int, exc: BaseException) -> None:
                if isinstance(exc, _EmptyResponse):
                    logger.info(
                        "empty response on attempt %d/%d", attempt, max_attempts
                    )
                else:
                    logger.warning(
                        "retry %d/%d in %s: %s",
                        attempt,
                        max_attempts,
                        method.__name__,
                        exc,
                    )

            try:
                return retry_with_backoff(
                    call_once,
                    # Guard against a non-positive configured attempt count,
                    # which would otherwise skip the call entirely.
                    max_attempts=max(1, cfg.chat_retry_max_attempts),
                    initial_delay=cfg.chat_retry_initial_delay,
                    multiplier=cfg.chat_retry_backoff_multiplier,
                    max_delay=cfg.chat_retry_max_delay,
                    on_error=on_error,
                )
            except _EmptyResponse:
                logger.error("retries exhausted in %s: empty response", method.__name__)
                return ""
            except Exception as exc:
                logger.error("retries exhausted in %s: %s", method.__name__, exc)
                return ""

        return wrapper

    if method is not None:
        return decorate(method)
    return decorate


class ConceptSampler:
    def __init__(self, model=None, model_name=None, port=8000, num_gpus=1, config=None):
        self.config = config or ConceptSamplerConfig()
        self.model, self.model_name = resolve_model(
            model,
            model_name,
            default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            port=port,
            num_gpus=num_gpus,
        )
        # State of the logprob filter. One ConceptSampler is shared by every
        # concurrent game, so the lock guards every field below.
        self._logprob_filter_lock = threading.Lock()
        self._logprob_filter_available = self.config.use_logprob_filter
        self._usable_logprob_replies = 0
        self._unusable_logprob_replies = 0
        if self.config.use_logprob_filter and is_reasoning_model(self.model_name):
            # These model families reject ``temperature`` and ``max_tokens``, and
            # their first token is not the answer.
            self._disable_logprob_filter(
                f"{self.model_name} is a reasoning model, so the first token is "
                "not the answer"
            )

    def latest_qa_pair(self, messages: list[dict], stage: int) -> tuple[str, str]:
        """Return the (question, answer) pair completed just before ``stage`` began.

        ``messages`` follows the ``[head, q1, a1, q2, a2, ...]`` layout that
        :func:`_qa_pairs` assumes. ``stage`` is 1-indexed and names the turn
        about to start, so the pair wanted is the ``(stage - 1)``-th completed
        turn, which is index ``stage - 2`` in the 0-indexed pair list.

        Raises ValueError when the requested pair does not exist.
        """
        pairs = _qa_pairs(messages)
        if stage < 2:
            raise ValueError(
                f"stage must be 2 or more; no turn has completed before stage {stage}"
            )
        if len(pairs) < stage - 1:
            raise ValueError(
                f"stage {stage} needs {stage - 1} completed turns, but the "
                f"transcript holds {len(pairs)}"
            )
        return pairs[stage - 2]

    def pre_filter(
        self,
        messages: list[dict],
        stage: int,
        valid_concepts: list[str],
    ) -> list[str]:
        """Filter the previously-valid concept pool by the latest Q/A only."""
        question, answer = self.latest_qa_pair(messages, stage)
        logger.info(
            "pre_filter Q=%s A=%s | concepts=%s", question, answer, valid_concepts
        )
        kept = self.check_samples_single_answer(
            question=question, answer=answer, concepts=valid_concepts, chat=True
        )
        logger.info("pre_filter kept %d/%d", len(kept), len(valid_concepts))
        return kept

    def build_sample_request(
        self,
        *,
        messages: list[dict],
        task: str,
        num_samples: int,
        stage: int,
        valid_samples: list[str],
        declarative: bool,
        declarative_messages: Optional[list[str]] = None,
    ) -> list[dict]:
        """Assemble the chat-completions ``messages`` payload."""
        strategy = _CONCEPT_PROMPTS[task]
        if stage == 1:
            content = strategy.initial(num_samples)
        else:
            content = strategy.iterative(num_samples)
            if valid_samples:
                content = (
                    content
                    + "\n"
                    + f"These examples have already been generated: {valid_samples}\n"
                    + "You should not repeat any of these examples. "
                )

        new_messages = self.reverse_pairs(messages)
        if stage == 1:
            new_messages = new_messages[1:]  # drop the head message
        new_messages.append({"role": "user", "content": content})
        if declarative:
            new_messages = [
                {
                    "role": "user",
                    "content": self.create_declarative_prompt_from_declarative(
                        declarative_messages or [], num_samples, reverse=True
                    )
                    if stage > 1
                    # At stage 1 this is the same initial prompt ``content``
                    # already holds, so it is reused rather than recomputed.
                    else content,
                }
            ]
        return new_messages

    def _parse_concepts(self, output: str) -> list[str]:
        """Strip ``<think>``, find the boundary, clean each line, drop noise.

        Shares the extraction skeleton with
        ``QuestionSampler.generate_followup_questions`` via
        ``utils.parse_llm_line_list``; only the per-line cleaning differs, and
        that lives in :func:`_clean_concept_line`.

        Run with ``PYTHONHASHSEED=0`` (see the README).
        """
        return list(
            set(
                parse_llm_line_list(
                    output,
                    boundary_prefixes=_BOUNDARY_PREFIXES,
                    clean_line=_clean_concept_line,
                )
            )
        )

    def post_filter(self, messages: list[dict], samples: list[str]) -> list[str]:
        """Stage > 1 multi-turn consistency filter."""
        return self.check_samples_multiple_answer(
            messages=messages, concepts=samples, chat=True
        )

    @chat_with_retries
    def _do_chat(
        self, request_messages: list[dict], num_samples: int, stage: int
    ) -> str:
        """One chat completion call.

        When ``stage`` is below ``multi_completion_stage_cutoff`` (endgame
        switch), the requested completions are concatenated to widen the
        candidate pool; at or above it only the first is used.
        """
        is_reasoning = is_reasoning_model(self.model_name)
        sampling_kwargs = (
            {}
            if is_reasoning
            else {
                "temperature": self.config.temperature,
                "max_tokens": num_samples * self.config.tokens_per_sample,
            }
        )
        response = self.model.client.chat.completions.create(
            model=self.model_name,
            messages=request_messages,
            n=self.config.n_completions,
            **sampling_kwargs,
        )
        # ``content`` is None (not "") whenever a reasoning parser consumes the
        # whole token budget, so coalesce rather than letting str.join raise a
        # TypeError that the retry wrapper would swallow into an empty pool.
        if stage < self.config.multi_completion_stage_cutoff:
            return "\n".join(
                (choice.message.content or "")
                for choice in response.choices[: self.config.n_completions]
            )
        return response.choices[0].message.content or ""

    def generate_diverse_samples_from_chat(
        self,
        messages,
        num_samples=None,
        stage=None,
        chat=True,
        declarative=False,
        declarative_messages=None,
        valid_concepts=None,
        task="celebrities",
        filter=True,
    ):
        if num_samples is None:
            num_samples = self.config.num_samples
        if declarative_messages is None:
            declarative_messages = []
        if valid_concepts is None:
            valid_concepts = []
        if stage is None:
            raise ValueError("stage is required and must be a 1-indexed int")
        if task not in _CONCEPT_PROMPTS:
            raise ValueError(f"Unsupported task: {task!r}")

        valid_samples: list[str] = []
        if valid_concepts and stage > 1 and filter:
            valid_samples = self.pre_filter(messages, stage, valid_concepts)

        effective_num_samples = (
            self.config.stage1_oversample_factor * num_samples
            if stage == 1
            else num_samples
        )

        new_messages = self.build_sample_request(
            messages=messages,
            task=task,
            num_samples=effective_num_samples,
            stage=stage,
            valid_samples=valid_samples,
            declarative=declarative,
            declarative_messages=declarative_messages,
        )

        samples: list[str] = []
        samples_before: list[str] = []
        iteration = 0
        while not samples and iteration < self.config.max_iterations:
            iteration += 1
            # ``samples_before`` is only populated by the stage > 1 filter,
            # so the guard keeps the retry prompt to non-empty example lists.
            if iteration > 1 and not declarative and samples_before:
                new_messages[-1]["content"] = (
                    new_messages[-1]["content"]
                    + "\n"
                    + "The previous examples were: "
                    + "\n".join(samples_before)
                    + "\n"
                    + "None of the examples above are correct. "
                )
            if chat:
                logger.info("chat request: %s", new_messages)
                output = self._do_chat(new_messages, effective_num_samples, stage)
            else:
                response = self.model.client.completions.create(
                    model=self.model_name,
                    prompt=format_conversation(messages) + new_messages[-1]["content"],
                )
                output = response.choices[0].text

            samples = self._parse_concepts(output)
            if stage > 1 and filter:
                # Cap on the pool handed to the consistency filter. Run with
                # PYTHONHASHSEED=0 for a fixed ordering (see the README).
                samples_before = samples[
                    : (effective_num_samples * self.config.filter_oversample_factor)
                ]
                logger.info("samples before filter: %s", samples_before)
                samples = self.post_filter(messages, samples_before)
                logger.info("samples after filter: %s", samples)
                samples = valid_samples + samples
                samples = list(set(samples))
                logger.info("samples after adding valid: %s", samples)

        if len(samples) < num_samples:
            logger.warning("only %d samples were generated", len(samples))

        return samples, declarative_messages

    # Deprecated.
    def ask_llm(self, message, concept) -> bool:
        warnings.warn(
            "ConceptSampler.ask_llm is deprecated and slated for removal.",
            DeprecationWarning,
            stacklevel=2,
        )
        user_messages = [
            {
                "role": "system",
                "content": f"Based on your knowledge about {concept}, "
                f"conclude whether it is consistent with this statement: {message}. "
                f"Limit your respond to only 'Yes.', or 'No.', with no explanation or other words. "
                f"Never say the answer {concept} in your response. ",
            },
            {
                "role": "user",
                "content": f"Does the person {concept} fit with the following statement?\n{message}\n"
                "Answer from (Yes/No)",
            },
        ]

        prompt = format_conversation(user_messages)

        prompt += "\nASSISTANT:<|START_RESPONSE|>"

        response = self.model.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=_YES_NO_MAX_TOKENS,
            logprobs=_YES_NO_TOP_LOGPROBS,
        )
        logprobs = response.choices[0].logprobs.top_logprobs[0]
        p_no = 0.0
        if "No" in logprobs:
            p_no = np.exp(logprobs["No"])
        # This method applies an absolute threshold to p(no). See
        # ``check_single_answer`` for the form that compares p(no) to p(yes).
        if p_no > self.config.consistency_threshold:
            return False

        return True

    # Deprecated.
    def convert_messages_to_declarative(self, messages, reverse=False):
        warnings.warn(
            "ConceptSampler.convert_messages_to_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        declarative_messages = []
        for question, answer in _qa_pairs(messages):
            qa_pair = f"Question: {question}, Answer: {answer}"
            declarative_qa = ""
            while declarative_qa == "":
                declarative_qa = self.convert_to_declarative(qa_pair)
            declarative_messages.append(declarative_qa)

        if reverse:
            declarative_messages = declarative_messages[::-1]
        logger.info("declarative messages count: %d", len(declarative_messages))
        return declarative_messages

    # Deprecated.
    def check_samples_declarative(self, declarative_messages, concepts) -> list:
        warnings.warn(
            "ConceptSampler.check_samples_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        for message in declarative_messages:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.ask_llm, message, concept)
                    for concept in concepts
                ]
                results = [future.result() for future in futures]
                concepts = [
                    concept for concept, valid in zip(concepts, results) if valid
                ]

            if not concepts:
                break

        return concepts

    # Deprecated.
    def convert_to_declarative(self, qa_pair):
        warnings.warn(
            "ConceptSampler.convert_to_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        prompt = (
            "Convert the following Q&A pair into a declarative statement.\n\n"
            "Example:\n"
            "Question: Is the sun pink or white?, Answer: No.\n"
            "Statement: The sun is not pink or white.\n\n"
            "Now, convert this Q&A pair:\n"
            f"{qa_pair}\n"
            "Statement:<|START_RESPONSE|>"
        )

        response = self.model.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=30,
            temperature=0.15,
            top_p=0.95,
            extra_body={
                "repetition_penalty": 1.1,
            },
        )
        response = response.choices[0].text
        response = re.sub(r"<\|END_RESPONSE\|>", "", response)
        return response.split(".")[0].lstrip().strip()

    # Deprecated.
    def create_declarative_prompt_from_declarative(
        self, declarative_messages, num_samples, reverse=False
    ):
        warnings.warn(
            "ConceptSampler.create_declarative_prompt_from_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        if reverse:
            declarative_messages = declarative_messages[::-1]
        statements = "\n".join(
            f"{i}. {message}" for i, message in enumerate(declarative_messages, 1)
        )

        prompt = (
            "You are participating in a game of 20 questions."
            f"Based on the information provided in the statements below, generate a list of **up to** {num_samples} diverse, unique examples of historical or current famous figures (real or fictional). Each person must be consistent with *all* the established clues from the conversation. "
            "Statements:\n"
            f"{statements}\n\n"
            "Each example should be on a separate line and be a single distinct entity with nothing else added (e.g., 'George Washington', 'Blaise Pascal', 'James Brown'), and no repetition. "
            "Ensure diversity in terms of time periods, geographic locations, and areas of achievement, while strictly adhering to the constraints set by the previous questions and answers. Some of the examples should be more obscure."
        )

        return prompt

    def reverse_pairs(self, messages):
        head = messages[0]

        pairs = list(zip(messages[1::2], messages[2::2]))

        reversed_pairs = pairs[::-1]

        result = [head] + [element for pair in reversed_pairs for element in pair]

        return result

    def check_single_answer(
        self,
        concept: str,
        question: str,
        answer: str,
        chat: bool = False,
    ) -> bool:
        # Reuse the answerer prompt so the consistency check conditions on
        # exactly the prompt that produced the answer being checked. NOTE: this
        # intentionally does not forward ``enable_maybe`` -- the filter is always
        # asked a strict Yes/No question, as it was before this was shared.
        user_messages = _PROMPT_BUILDER.answerer_messages(concept, question)
        if not chat:
            prompt = format_conversation(user_messages)

            prompt += "\nASSISTANT:<|START_RESPONSE|>"

            response = self.model.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=_YES_NO_MAX_TOKENS,
                logprobs=_YES_NO_TOP_LOGPROBS,
                temperature=_GREEDY_TEMPERATURE,
            )
            logprobs = response.choices[0].logprobs.top_logprobs[0]
        else:
            expected = _yes_no_polarity(answer)
            if expected is None:
                return True
            # The threshold rule needs the same prompt and the same model as the
            # text check, so it runs here and replaces the reply comparison.
            if self._logprob_filter_enabled():
                log_probs = self._answer_token_logprobs(user_messages)
                if log_probs is not None:
                    return self._survives_logprob_filter(
                        log_probs, expected, concept, question
                    )
            output = self._check_single_answer_chat(user_messages)
            if not output:
                return True
            # Reasoning models can emit chain-of-thought into ``content``; keep
            # only the post-</think> text, as ``_parse_concepts`` does.
            output = output.split("</think>")[-1]
            return _yes_no_polarity(output) == expected
        if self._logprob_filter_enabled() and has_yes_no_mass(logprobs):
            expected = _yes_no_polarity(answer)
            if expected is None:
                return True
            return self._survives_logprob_filter(logprobs, expected, concept, question)
        p_no = 0.0
        p_yes = 0.0
        if "Yes" in logprobs:
            p_yes = np.exp(logprobs["Yes"])
        if "No" in logprobs:
            p_no = np.exp(logprobs["No"])

        if answer.lower().startswith("yes"):
            if p_no > p_yes:
                return False
        if answer.lower().startswith("no"):
            if p_yes > p_no:
                return False

        return True

    def _logprob_filter_enabled(self) -> bool:
        """Return True while the logprob filter can still run this run."""
        return self.config.use_logprob_filter and self._logprob_filter_available

    def _survives_logprob_filter(
        self,
        log_probs: dict,
        expected: bool,
        concept: str,
        question: str,
    ) -> bool:
        """Keep the concept unless the model is confident it contradicts the answer.

        ``log_probs`` holds the answer-token log-probabilities for one
        (concept, question) pair. ``extract_yes_no_probs`` renormalises them
        over {yes, no}, which is the same p(a | question, concept) that the EIG
        integrates, so the filter and the acquisition read one quantity.

        ``expected`` is the polarity of the answer the game observed. The other
        polarity is the contradiction, and its probability is how confident the
        model is that this concept is wrong. The rule rejects the concept when
        that confidence goes above ``logprob_reject_threshold``; a tie at the
        threshold keeps it. 0.5 is the neutral point and matches the text
        check, and below 0.5 is stricter. The README covers the threshold
        semantics and the differences from the text check.

        The rule applies once per Q/A pair, so a concept survives a multi-turn
        sweep only when no pair rejects it (``check_samples_multiple_answer``).
        A joint score over the pairs is deliberately NOT used: the pairs are
        separate constraints, and a product of per-pair probabilities shrinks as
        the game gets longer, so it would drop a fully consistent concept late
        in a game.
        """
        p_yes, p_no = extract_yes_no_probs(log_probs)
        p_contradiction = p_no if expected else p_yes
        # ``bool()`` because the probabilities are numpy floats, so the bare
        # comparison returns numpy.bool_ and the annotation would not hold.
        keep = bool(p_contradiction <= self.config.logprob_reject_threshold)
        logger.debug(
            "logprob filter: concept=%s answer=%s p(contradiction)=%.4f "
            "threshold=%.2f keep=%s | Q=%s",
            concept,
            "yes" if expected else "no",
            p_contradiction,
            self.config.logprob_reject_threshold,
            keep,
            question,
        )
        return keep

    def _answer_token_logprobs(self, user_messages: list[dict]) -> Optional[dict]:
        """Return {token: logprob} for the answer token, or None.

        None means the reply carries no usable distribution: the request
        failed, the backend sent no logprobs, or no position held enough yes/no
        mass. The caller then falls back to the text check for this concept, so
        a backend without chat logprobs keeps the filter working instead of
        turning it into a no-op. ``_record_logprob_reply`` decides when the
        evidence justifies switching the mode off for the whole run; a
        transient failure is not counted as evidence.

        The scan takes the first returned position that holds yes/no mass. At
        the shipped one-token budget there is only one position.
        """
        response = self._answer_logprobs_chat(user_messages)
        if response is None:
            self._record_logprob_reply(usable=False)
            return None
        if not response:
            logger.info("the logprob request failed; using the text check this time")
            return None
        choice_logprobs = getattr(response.choices[0], "logprobs", None)
        for entry in getattr(choice_logprobs, "content", None) or []:
            log_probs = top_logprob_dict(getattr(entry, "top_logprobs", None) or [])
            if has_yes_no_mass(log_probs):
                self._record_logprob_reply(usable=True)
                return log_probs
        self._record_logprob_reply(usable=False)
        return None

    @chat_with_retries(retry_on_empty=False)
    def _answer_logprobs_chat(self, user_messages: list[dict]):
        """One chat call that returns the answer-token log-probabilities.

        Mirrors ``q20_llm_io.DefaultQ20LlmInterface._yes_no_probs_via_chat``:
        the same prompt, the same guesser model, greedy decoding, and one token.
        The two must stay the same, because the filter thresholds the
        distribution that the EIG integrates. One difference remains:
        ``--history`` prepends the transcript to the EIG prompt and never to
        this one, because the filter checks one Q/A pair.

        A rejected request returns None instead of raising, so the retry loop
        does not sleep through its whole policy for an option the backend will
        never support. The caller reads None as "use the text check".

        The one-token budget targets models that answer directly. A reasoning
        model that emits a think token first is reported as undecidable, and the
        caller then uses the text check.
        """
        try:
            return self.model.client.chat.completions.create(
                model=self.model_name,
                messages=user_messages,
                logprobs=True,
                top_logprobs=_FILTER_TOP_LOGPROBS,
                temperature=_GREEDY_TEMPERATURE,
                stream=False,
                max_tokens=_FILTER_LOGPROB_MAX_TOKENS,
            )
        except Exception as exc:
            if _is_unsupported_request(exc):
                # Logged at debug because every in-flight thread reaches this line
                # before the latch is set. ``_disable_logprob_filter`` carries the
                # one warning the run needs.
                logger.debug("the API rejected the request for logprobs: %s", exc)
                return None
            raise

    def _disable_logprob_filter(self, reason: str) -> None:
        """Switch the logprob filter off for the rest of the run, and say why once."""
        with self._logprob_filter_lock:
            if not self._logprob_filter_available:
                return
            self._logprob_filter_available = False
        logger.warning(
            "logprob filter off for the rest of the run (%s); the consistency "
            "filter uses the text yes/no check",
            reason,
        )

    def _record_logprob_reply(self, usable: bool) -> None:
        """Count one reply, and switch the mode off when the evidence is clear.

        The rule needs both a sample count and a high share of unusable
        replies, so neither one bad reply nor a chance run of them changes the
        method for a whole run. The counters are cumulative, not consecutive,
        because every game shares them and thread completion order carries no
        information about the backend.
        """
        with self._logprob_filter_lock:
            if usable:
                self._usable_logprob_replies += 1
                return
            self._unusable_logprob_replies += 1
            unusable = self._unusable_logprob_replies
            total = unusable + self._usable_logprob_replies
            switch_off = (
                unusable >= _LOGPROB_FILTER_MIN_SAMPLES
                and unusable / total >= _LOGPROB_FILTER_MAX_UNUSABLE_SHARE
            )
        if switch_off:
            self._disable_logprob_filter(
                f"{unusable} of {total} replies carried no usable yes/no logprobs"
            )

    @chat_with_retries(retry_on_empty=False)
    def _check_single_answer_chat(self, user_messages: list[dict]) -> Optional[str]:
        is_reasoning = is_reasoning_model(self.model_name)
        response = self.model.client.chat.completions.create(
            model=self.model_name,
            messages=user_messages,
            **(
                {}
                if is_reasoning
                else {
                    "temperature": _GREEDY_TEMPERATURE,
                    "max_tokens": self.config.chat_answer_max_tokens,
                }
            ),
        )
        return response.choices[0].message.content

    def _filter_concepts_by_qa(
        self,
        question: str,
        answer: str,
        concepts: list[str],
        chat: bool,
        executor: ThreadPoolExecutor,
    ) -> list[str]:
        """Keep the concepts consistent with one Q/A pair, checked in parallel."""
        futures = [
            executor.submit(self.check_single_answer, concept, question, answer, chat)
            for concept in concepts
        ]
        results = [future.result() for future in futures]
        return [concept for concept, valid in zip(concepts, results) if valid]

    def check_samples_single_answer(
        self,
        question: str,
        answer: str,
        concepts: list[str],
        chat: bool = False,
    ) -> list[str]:
        if not concepts:
            return []
        with ThreadPoolExecutor(max_workers=_filter_max_workers(concepts)) as executor:
            return self._filter_concepts_by_qa(
                question, answer, concepts, chat, executor
            )

    def check_samples_multiple_answer(
        self,
        messages: list[dict],
        concepts: list[str],
        chat: bool = False,
    ) -> list[str]:
        if not concepts:
            return []
        # One executor for the whole sweep rather than a fresh pool per Q/A pair.
        with ThreadPoolExecutor(max_workers=_filter_max_workers(concepts)) as executor:
            for question, answer in _qa_pairs(messages):
                concepts = self._filter_concepts_by_qa(
                    question, answer, concepts, chat, executor
                )
                if not concepts:
                    break
        return concepts

    # Deprecated.
    def rank_concepts(self, concepts, messages, item):
        warnings.warn(
            "ConceptSampler.rank_concepts is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        user_messages = [
            {
                "role": "user",
                "content": f"I have the following list of possible answers:\n{concepts}\n. Please rank the top 10 possible answers in order of likelihood, with the most likely answer first. "
                "Please provide the ranking in a numbered list format, with each answer on a separate line. Preserve the original text of each answer."
                "Please do not provide any explanations or additional information, just the ranking.",
            },
        ]

        new_messages = messages.copy()
        new_messages.append(user_messages[0])

        response = self.model.client.chat.completions.create(
            model=self.model_name,
            messages=new_messages,
            temperature=0.7,
            n=1,
        )
        logger.info("ranking: %s", response.choices[0].message.content)
        output = response.choices[0].message.content
        rank = score_ranked_list(output, item)
        return rank


if __name__ == "__main__":
    concept_sampler = ConceptSampler(
        model_name="CohereForAI/c4ai-command-a-03-2025", num_gpus=4
    )

    # The transcript layout the class assumes is [head, q1, a1, q2, a2, ...] --
    # an odd-length list whose first element is not part of a Q/A pair. A
    # head-less (even-length) list makes ``reverse_pairs`` silently drop the
    # last message, so the head is required here.
    chat_messages = [
        {"role": "user", "content": "Let's play 20 questions."},
        {"role": "assistant", "content": "Is it a living thing?"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "content": "Is it a plant?"},
        {"role": "user", "content": "No."},
    ]

    samples, declarative_messages = concept_sampler.generate_diverse_samples_from_chat(
        chat_messages, num_samples=25, stage=3, task="animals"
    )

    for idx, sample in enumerate(samples, start=1):
        print(f"{idx}. {sample}")
