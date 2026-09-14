import logging
import re
import time
import warnings
from dataclasses import dataclass

from retry_utils import retry_with_backoff
from serve_model import is_reasoning_model, resolve_model
from utils import (
    format_conversation,
    get_animal_question_prompt,
    get_celebrity_question_prompt,
    item_probs_from_completion_logprobs,
    get_thing_question_prompt,
    parse_llm_line_list,
)

logger = logging.getLogger(__name__)

# Question-generation prompt builders keyed by dataset. Mirrors the
# ``_CONCEPT_PROMPTS`` registry in concept_sampler.py so that both samplers
# expose the same set of datasets in the same shape; every builder takes
# ``(num_questions, concepts, final)``.
_QUESTION_PROMPTS = {
    "celebrities": get_celebrity_question_prompt,
    "animals": get_animal_question_prompt,
    "things": get_thing_question_prompt,
}

# Everything up to and including the first line starting with one of these is
# treated as preamble and dropped by ``parse_llm_line_list``.
_QUESTION_BOUNDARY_PREFIXES = ("Based",)

# Response cue appended to a flat transcript before asking for completion
# logprobs. Matches the ``Role: content`` casing that ``format_conversation``
# emits directly above it.
_RESPONSE_CUE = "\nAssistant:<|START_RESPONSE|>"

# Attempts for a single chat/completions request. These bound one API call;
# ``QuestionSamplerConfig.max_attempts`` separately bounds how many times the
# whole generate-and-parse cycle is retried.
_CHAT_MAX_ATTEMPTS = 10
_COMPLETION_MAX_ATTEMPTS = 50

# Backoff policy for one request, in seconds. The chat path doubles the delay.
# The completions path holds it flat, because a multiplier of 1.0 reproduces the
# fixed ``time.sleep(1.0)`` loop that this call replaced. The cap matches
# ``ConceptSamplerConfig.chat_retry_max_delay``, so no single sleep can outlast a
# server restart. Without it, attempt 10 of the chat path sleeps 256 s.
_RETRY_INITIAL_DELAY = 1.0
_CHAT_RETRY_MULTIPLIER = 2.0
_COMPLETION_RETRY_MULTIPLIER = 1.0
_RETRY_MAX_DELAY = 128.0

# Delay before the whole generate-and-parse cycle starts again. It applies after
# the inner attempts above are exhausted.
_OUTER_RETRY_DELAY = 1.0

# Cap for the deprecated declarative conversion, which retries until the model
# returns a non-empty statement.
_DECLARATIVE_MAX_ATTEMPTS = 10

# Bullet marker that the question prompts ask the model not to emit, and that it
# emits anyway. Stripped from the front of a line only.
_BULLET_PREFIX = "- "


def _clean_question_line(line):
    """Strip role/numbering prefixes and response sentinels from one line."""
    cleaned_line = re.sub(r"^(Assistant:|User:|\d+\.\s*)", "", line).strip()
    cleaned_line = re.sub(r"<\|END_RESPONSE\|>", "", cleaned_line).strip()
    cleaned_line = cleaned_line.removeprefix(_BULLET_PREFIX)
    return cleaned_line


def _split_item_aliases(item):
    """Split a ``"a | b"`` dataset item into stripped alias strings.

    ``utils.text_mentions_item`` strips each alternative before matching, so the
    logprob path must strip too: scoring a raw ``" Joanne Rowling"`` would
    tokenize the leading space and misalign every ``top_logprobs`` lookup.
    """
    return [alias.strip() for alias in item.split("|")]


@dataclass
class QuestionSamplerConfig:
    """Hyperparameters for question generation."""

    num_questions: int = 10
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    max_attempts: int = 2
    declarative_temperature: float = 0.15
    declarative_top_p: float = 0.95
    declarative_max_tokens: int = 30
    declarative_repetition_penalty: float = 1.1


class QuestionSampler:
    def __init__(self, model=None, model_name=None, port=8000, num_gpus=1, config=None):
        self.config = config or QuestionSamplerConfig()
        self.model, self.model_name = resolve_model(
            model,
            model_name,
            default="mistralai/Mistral-Small-24B-Instruct-2501",
            port=port,
            num_gpus=num_gpus,
        )

    def _supports_completion_logprobs(self) -> bool:
        """True when the backend is a locally served vLLM model.

        Hosted OpenAI models are excluded because ``/v1/completions`` with
        ``echo`` + ``logprobs`` is unavailable there and, for the reasoning
        families, ``AutoTokenizer.from_pretrained(model_name)`` has no HF repo
        to load. Mirrors ``Q20._is_api_model`` rather than testing
        ``startswith("gpt")``, which would admit o1/o3/o4.
        """
        return not (
            self.model_name.startswith("gpt") or is_reasoning_model(self.model_name)
        )

    def _correct_guess_logprob(self, prompt, possible_responses):
        """Log and return the per-alias logprob of the true item."""
        answer_prob = item_probs_from_completion_logprobs(
            prompt=prompt + _RESPONSE_CUE,
            possible_responses=possible_responses,
            model=self.model,
            logprobs=True,
        )
        logger.info("Probability of correct guess: %s", answer_prob)
        return answer_prob

    def generate_followup_questions(
        self,
        messages,
        num_questions=None,
        final=False,
        chat=True,
        concepts=None,
        declarative=False,
        declarative_messages=None,
        dataset="celebrities",
        item=None,
        return_answer_probs=False,
        log_correct_guess_prob: bool = True,
    ):
        concepts = concepts if concepts is not None else []
        declarative_messages = (
            declarative_messages if declarative_messages is not None else []
        )
        if num_questions is None:
            num_questions = self.config.num_questions
        if num_questions <= 0:
            raise ValueError(f"num_questions must be positive, got {num_questions!r}")
        if dataset not in _QUESTION_PROMPTS:
            raise ValueError(f"Unsupported dataset: {dataset!r}")
        build_prompt = _QUESTION_PROMPTS[dataset]

        answer_prob = None
        last_question = messages[-2]["content"].strip() if len(messages) >= 2 else None
        if len(concepts) == 1 and last_question != f"Is it {concepts[0]}?":
            if log_correct_guess_prob and self._supports_completion_logprobs():
                # Same conditioning context and the same dataset-specific guess
                # prompt as the every-turn probe below, so both feed one
                # comparable ``correct_guess_logprob`` series.
                new_messages = messages.copy()
                new_messages.append(
                    {
                        "role": "user",
                        "content": build_prompt(num_questions, concepts, final=True),
                    }
                )
                answer_prob = self._correct_guess_logprob(
                    format_conversation(new_messages),
                    _split_item_aliases(item) if item is not None else concepts,
                )
            if return_answer_probs:
                return [f"Is it {concepts[0]}?"], answer_prob
            return [f"Is it {concepts[0]}?"]

        question_gen_prompt = build_prompt(num_questions, concepts, final=final)

        new_message = {"role": "user", "content": question_gen_prompt}
        new_messages = messages.copy()
        new_messages.append(new_message)
        conversation = format_conversation(new_messages)
        if (
            log_correct_guess_prob
            and item is not None
            and self._supports_completion_logprobs()
        ):
            answer_prob = self._correct_guess_logprob(
                conversation, _split_item_aliases(item)
            )

        if chat and declarative:
            # Loop-invariant, so built once instead of on every retry.
            new_messages = [
                {
                    "role": "user",
                    "content": self.create_declarative_prompt_from_declarative(
                        declarative_messages=declarative_messages,
                        num_samples=num_questions,
                        concepts=concepts,
                        final=final,
                    ),
                }
            ]

        questions = []
        outer_attempts = 0
        # Loop-invariant: depends only on the model name and the config.
        sampling_kwargs = (
            {}
            if is_reasoning_model(self.model_name)
            else {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
        )
        while not questions:
            outer_attempts += 1
            if outer_attempts > self.config.max_attempts:
                raise Exception(
                    "Failed to generate questions after "
                    f"{self.config.max_attempts} attempts"
                )
            if chat:
                try:
                    response = retry_with_backoff(
                        lambda: self.model.client.chat.completions.create(
                            model=self.model_name,
                            messages=new_messages,
                            **sampling_kwargs,
                        ),
                        max_attempts=_CHAT_MAX_ATTEMPTS,
                        initial_delay=_RETRY_INITIAL_DELAY,
                        multiplier=_CHAT_RETRY_MULTIPLIER,
                        max_delay=_RETRY_MAX_DELAY,
                        on_error=lambda attempt, total, exc: logger.warning(
                            "Error in question sampler (chat) attempt %s/%s: %s",
                            attempt,
                            total,
                            exc,
                        ),
                    )
                except Exception as e:
                    logger.warning("Error in question sampler: %s", e)
                    time.sleep(_OUTER_RETRY_DELAY)
                    continue
                # ``content`` is None (not "") whenever a reasoning parser
                # consumes the whole token budget, so coalesce rather than
                # letting the empty generation raise out of the parser.
                output = response.choices[0].message.content or ""
            else:
                try:
                    response = retry_with_backoff(
                        lambda: self.model.client.completions.create(
                            model=self.model_name,
                            prompt=conversation,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            extra_body={
                                "repetition_penalty": self.config.repetition_penalty,
                            },
                        ),
                        max_attempts=_COMPLETION_MAX_ATTEMPTS,
                        initial_delay=_RETRY_INITIAL_DELAY,
                        multiplier=_COMPLETION_RETRY_MULTIPLIER,
                        max_delay=_RETRY_MAX_DELAY,
                        on_error=lambda attempt, total, exc: logger.warning(
                            "Error in question sampler (completions) attempt %s/%s: %s",
                            attempt,
                            total,
                            exc,
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        "Error in question sampler (completions outer): %s", e
                    )
                    time.sleep(_OUTER_RETRY_DELAY)
                    continue
                # A ``/v1/completions`` choice carries ``text``; only a chat
                # choice has ``message``.
                output = response.choices[0].text or ""

            questions = parse_llm_line_list(
                output,
                boundary_prefixes=_QUESTION_BOUNDARY_PREFIXES,
                clean_line=_clean_question_line,
            )

        if len(questions) < num_questions:
            logger.warning(
                "Only %s questions generated. Consider re-running or refining the prompt.",
                len(questions),
            )
        questions = questions[:num_questions]

        if return_answer_probs:
            return questions, answer_prob
        return questions

    # Deprecated.
    def convert_messages_to_declarative(self, messages, reverse=False):
        warnings.warn(
            "QuestionSampler.convert_messages_to_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        declarative_messages = []
        for i in range(1, len(messages) - 1, 2):
            question = messages[i]["content"].strip()
            answer = messages[i + 1]["content"].strip()
            qa_pair = f"Question: {question}, Answer: {answer}"
            declarative_qa = ""

            for _ in range(_DECLARATIVE_MAX_ATTEMPTS):
                declarative_qa = self.convert_to_declarative(qa_pair)
                if declarative_qa:
                    break
            else:
                logger.warning(
                    "no declarative statement after %s attempts for: %s",
                    _DECLARATIVE_MAX_ATTEMPTS,
                    qa_pair,
                )
            declarative_messages.append(declarative_qa)

        if reverse:
            declarative_messages = declarative_messages[::-1]

        return declarative_messages

    # Deprecated.
    def convert_to_declarative(self, qa_pair):
        warnings.warn(
            "QuestionSampler.convert_to_declarative is deprecated.",
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
            "Statement:"
        )

        response = self.model.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.config.declarative_max_tokens,
            temperature=self.config.declarative_temperature,
            top_p=self.config.declarative_top_p,
            extra_body={
                "repetition_penalty": self.config.declarative_repetition_penalty,
            },
        )

        text = re.sub(r"<\|END_RESPONSE\|>", "", response.choices[0].text)
        return text.split(".")[0].lstrip().strip()

    # Deprecated.
    # ConceptSampler's method of the same name has a different signature.
    def create_declarative_prompt_from_declarative(
        self, declarative_messages, num_samples, concepts, final=False
    ):
        warnings.warn(
            "QuestionSampler.create_declarative_prompt_from_declarative is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )

        statements = "\n".join(
            f"{i}. {message}" for i, message in enumerate(declarative_messages, 1)
        )

        if final:
            prompt = (
                f"{statements}\n\n"
                f"There are {len(declarative_messages)} statements above. Deduce the identity of the famous person who is consistent with all of them. "
                f"Given the statements, another LLM has generated a possibly incomplete list of possible identities: {concepts} "
                "Based on the statements above, guess the identity of the person by asking a question which is answerable in a binary fashion ('i.e. is it John Doe?'). "
                "Just respond with the identity question and nothing else."
            )
        else:
            prompt = (
                f"{statements}\n\n"
                f"You are participating in a game of 20 questions to deduce the identity of a famous person (real or fictional). Based on the statements above, generate a list of **up to** {num_samples} diverse, strategic and insightful follow-up questions that will help efficiently narrow down the identity of the person."
                f"Given the questions and answers asked, another LLM has generated a list of possible identities: {concepts}. It may be good to try to narrow these down."
                "The questions must be strictly binary - each question can only be answered with 'Yes' or 'No'. Additionally, some of the questions should attempt to guess the identity "
                "of the item (for example, 'Is it item X?') while still being answerable in a binary fashion. "
                "Return each question on a separate line with no extra commentary or numbering."
                "Each question must be consistent with all statements above and should aim to maximize information gain by distinguishing between the remaining possibilities. Avoid redundancy and ensure that the questions cover diverse attributes such as time period, profession, nationality, and notable achievements."
                "If you are confident about the identity of the person, then just generate one question with your guess given all previous information."
            )
        return prompt


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Is it a living thing?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Is it a plant?"},
        {"role": "assistant", "content": "No."},
    ]

    question_sampler = QuestionSampler(model_name="google/gemma-3-27b-it")
    try:
        followup_questions = question_sampler.generate_followup_questions(
            messages, num_questions=15
        )
        for idx, q in enumerate(followup_questions, start=1):
            print(f"{idx}. {q}")
    finally:
        question_sampler.model.terminate()
