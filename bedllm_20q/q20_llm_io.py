import math
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from theta_eig_completions import item_probs_from_completion_logprobs
from utils import PROMPT, format_conversation


# Floor applied to a yes/no token that the backend did not report. Kept as the
# default of ``extract_yes_no_probs`` so the acquisition path and the consistency
# filter share one value.
DEFAULT_YES_NO_EPSILON = 1e-6


# Token spellings that carry yes/no answer mass, grouped by the answer they
# spell. A tokenizer can keep a word-start marker in the token string, so one
# answer reaches the top-k list under more than one spelling. Add a spelling to
# the group it belongs to and every reader below picks it up. Adding one also
# moves every recorded EIG number, because the acquisition functions consume
# the renormalised pair.
YES_SPELLINGS: tuple[str, ...] = ("Yes", "▁Yes")
NO_SPELLINGS: tuple[str, ...] = ("No", "▁No")

# Flat view for callers that weigh a reply without splitting it by answer.
YES_NO_TOKENS: tuple[str, ...] = YES_SPELLINGS + NO_SPELLINGS

# Share of the first token that must sit on the yes/no spellings before a caller
# may renormalise over them. The answerer prompt asks for "Yes." or "No.", so a
# model that follows it puts nearly all of the mass there. A reply far below this
# floor is answering some other way, and its renormalised ratio says more about
# which spelling reached the top-k list than about the answer.
MIN_YES_NO_MASS = 0.05


def top_logprob_dict(top_logprobs) -> Dict[str, float]:
    """Flatten one chat-completions ``top_logprobs`` entry into {token: logprob}."""
    return {
        token_logprob.token: token_logprob.logprob for token_logprob in top_logprobs
    }


def yes_no_mass(log_probs: Dict[str, float]) -> float:
    """Return the probability the reply puts on the yes/no tokens, before renormalising.

    Only a finite number counts. A backend that reports -inf, a large negative
    sentinel, or a non-number contributes nothing.
    """
    total = 0.0
    for token in YES_NO_TOKENS:
        value = log_probs.get(token)
        if isinstance(value, (int, float)) and math.isfinite(value):
            total += float(np.exp(value))
    return total


def has_yes_no_mass(
    log_probs: Dict[str, float], minimum: float = MIN_YES_NO_MASS
) -> bool:
    """Return True when the reply puts enough mass on a yes/no token to threshold.

    ``extract_yes_no_probs`` divides by the yes/no total, so it reports a ratio
    even when that total is nearly zero. Three untrustworthy replies look the
    same to it: a top-k window with no yes/no token and a window whose yes/no
    tokens are dead both give the (0.5, 0.5) of the two epsilon floors, and a
    window with one yes/no token at a tiny probability inverts the ratio,
    because the absent token's epsilon floor can exceed the present token's
    probability. That third case is the dangerous one — the number looks
    decisive. A reasoning model hits the first case on every reply, since its
    first token opens the chain of thought. Requiring real mass rejects all
    three.
    """
    return yes_no_mass(log_probs) >= minimum


def _answer_prob(
    log_probs: Dict[str, float], spellings: tuple[str, ...], epsilon: float
) -> float:
    """Return the mass one answer holds, summed over its spellings.

    An answer that no spelling in the top-k list reports contributes
    ``epsilon`` rather than zero, which keeps the ratio finite when only the
    other answer appears. The floor is per answer, so a single reported
    spelling replaces it.
    """
    total = 0.0
    reported = False
    for token in spellings:
        if token in log_probs:
            total += np.exp(log_probs[token])
            reported = True
    return total if reported else epsilon


def extract_yes_no_probs(
    log_probs: Dict[str, float], epsilon: float = DEFAULT_YES_NO_EPSILON
):
    """Return ``(p_yes, p_no)`` renormalised over the yes/no tokens only.

    ``log_probs`` maps a token string to its log-probability, as returned by
    ``top_logprob_dict`` or by the completions ``top_logprobs[0]`` field. The
    function reads the first answer token only, so the pair is the answerer's
    predictive distribution p(a | question, concept) restricted to {yes, no}.
    Both the acquisition functions and the consistency filter call this, so the
    two cannot drift apart.

    Each answer is summed over ``YES_SPELLINGS`` / ``NO_SPELLINGS``, and
    ``_answer_prob`` holds the epsilon rule. A backend reporting -inf for both
    answers leaves a total of 0.0, and the division would yield NaN, which wins
    the acquisition argmax and makes ``wandb.Histogram`` raise, so that case
    returns ``(0.5, 0.5)`` — no information. One dead answer still gives a
    decisive pair: ``{"Yes": -inf}`` returns ``(0.0, 1.0)``.
    """
    p_yes = _answer_prob(log_probs, YES_SPELLINGS, epsilon)
    p_no = _answer_prob(log_probs, NO_SPELLINGS, epsilon)
    total = p_yes + p_no
    if not total > 0:
        return 0.5, 0.5
    return p_yes / total, p_no / total


class Q20PromptBuilder:
    def answerer_messages(
        self, concept: str, question: str, enable_maybe: bool = False
    ) -> List[Dict[str, str]]:
        maybe_system = " or 'Don't Know.'" if enable_maybe else ""
        maybe_user = "/Don't know." if enable_maybe else ""
        return [
            {
                "role": "system",
                "content": (
                    f"Based on your knowledge about {concept}, "
                    f"respond to the following question or guess. "
                    f"Limit your response to only 'Yes.', or 'No.'{maybe_system}, with no explanation or other words. "
                    f"Never say the answer {concept} in your response. "
                    "Example:\n"
                    "For the person Angelina Jolie, Is the person Angelina Jolie?\n"
                    "Assistant: Yes.\n"
                ),
            },
            {
                "role": "user",
                "content": f"For {concept}, {question} (Yes/No{maybe_user})",
            },
        ]

    @staticmethod
    def history_answer_messages(
        messages: List[Dict[str, str]], question: str
    ) -> List[Dict[str, str]]:
        return messages + [
            {
                "role": "user",
                "content": (
                    "Based on the dialogue history, "
                    "respond to the following question or guess. "
                    "Limit your response to only 'Yes.', or 'No.', with no explanation or other words. "
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

    @staticmethod
    def answerer_logprob_messages(item: str, question: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    f"Based on your knowledge about {item}, "
                    f"respond to the following question or guess. "
                    f"Limit your respond to only 'Yes.', 'No.' or 'Maybe.', with no explanation or other words. "
                    f"Never say the answer {item} in your response. "
                    f"If the question is to solicit the answer, respond 'No.'."
                ),
            },
            {
                "role": "user",
                "content": f"For the entity {item}, {question} (Yes/No/Maybe)",
            },
        ]

    @staticmethod
    def answer_probability_prompt(
        messages: List[Dict[str, str]], prompt_suffix: str = PROMPT
    ) -> str:
        message_text = format_conversation(messages)
        prompt = (
            "Here is the conversation history with the user:\n"
            + message_text
            + "\n"
            + prompt_suffix
        )
        return prompt + "\nASSISTANT:<|START_RESPONSE|>"


class Q20LlmInterface(Protocol):
    def guesser_chat(self, messages=None, greedy_decode: bool = False): ...

    def yes_no_probs(
        self, question: str, concept: str, chat: bool = True, use_history: bool = False
    ): ...

    def answerer_log_probs(
        self,
        question: str,
        print_log_probs: bool = False,
        print_index_probs: bool = False,
    ): ...

    def answer_probs_given_history(self, question: str, chat: bool = True): ...

    def answerer(
        self, question: str, chat: bool = True, history=None, sample: bool = False
    ): ...

    def ask_llm(
        self, question: str, concept: str, chat: bool = True, seed: Optional[int] = None
    ): ...

    def get_answer_probabilities(self): ...

    def select_question_no_beliefs(self, followup_questions): ...

    def select_question_with_beliefs(
        self,
        candidate_questions,
        history=None,
        temperature: float = 0.7,
        explain: bool = False,
        concepts=None,
    ): ...


class DefaultQ20LlmInterface:
    def __init__(
        self,
        *,
        guesser_model: Any,
        answerer_model: Any,
        item: str,
        enable_maybe: bool,
        get_guesser_messages: Callable[[], List[Dict[str, str]]],
        retry: Callable[..., Any],
        is_reasoning_model: Callable[[str], bool],
        is_answer_revealed: Callable[[str], bool],
        epsilon: float = 1e-6,
        fuzzy_match_threshold: float = 0.8,
        prompt_builder: Optional[Q20PromptBuilder] = None,
        prompt_suffix: str = PROMPT,
    ):
        self.guesser_model = guesser_model
        self.answerer_model = answerer_model
        self.item = item
        self.enable_maybe = enable_maybe
        self._get_guesser_messages = get_guesser_messages
        self._retry = retry
        self._is_reasoning_model = is_reasoning_model
        self._is_answer_revealed = is_answer_revealed
        self.epsilon = epsilon
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.prompt_builder = (
            prompt_builder if prompt_builder is not None else Q20PromptBuilder()
        )
        self.prompt_suffix = prompt_suffix

    def _extract_yes_no_probs(self, log_probs: Dict[str, float]):
        return extract_yes_no_probs(log_probs, epsilon=self.epsilon)

    @staticmethod
    def _top_logprob_dict(top_logprobs) -> Dict[str, float]:
        return top_logprob_dict(top_logprobs)

    def _yes_no_probs_via_chat(
        self,
        model: Any,
        messages: List[Dict[str, str]],
        top_logprobs: int,
        context: str,
    ) -> Dict[str, float]:
        def _call():
            completion = model.client.chat.completions.create(
                model=model.model_name,
                messages=messages,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=0.0,
                stream=False,
                # The code below reads the top_logprobs of the FIRST token only.
                max_tokens=1,
            )
            lp = self._top_logprob_dict(
                completion.choices[0].logprobs.content[0].top_logprobs
            )
            if not lp:
                raise Exception("Empty logprobs returned")
            return lp

        return self._retry(_call, context=context)

    @staticmethod
    def _clean_completion_response(text: str):
        text = (
            text.split("</think>")[-1]
            .split("User")[0]
            .split("\n")[0]
            .split("<")[0]
            .lstrip()
            .strip()
        )
        text = re.sub(
            r"<\|END_RESPONSE\|>|<\|END_RESPONSE\||\|<\|END_RESPONSE", "", text
        )
        text = re.sub(r"<EOS_TOKEN>", "", text)
        return text

    def guesser_chat(self, messages=None, greedy_decode: bool = False):
        def _call():
            gen = self.guesser_model.client.chat.completions.create(
                model=self.guesser_model.model_name,
                messages=self._get_guesser_messages() if messages is None else messages,
                temperature=1.0 if not greedy_decode else 0.0,
            )
            gen_str = gen.choices[0].message.content.lstrip().strip()
            match = re.search(r"<question>(.*?)</question>", gen_str, re.DOTALL)
            if match:
                gen_str = match.group(1).strip()
                if not gen_str.endswith("?"):
                    gen_str += "?"
            return {"role": "assistant", "content": gen_str}

        return self._retry(_call, context="guesser_chat")

    def yes_no_probs(
        self, question: str, concept: str, chat: bool = True, use_history: bool = False
    ):
        user_messages = self.prompt_builder.answerer_messages(concept, question)
        if use_history:
            history = (
                self._get_guesser_messages()[1:]
                if len(self._get_guesser_messages()) > 1
                else []
            )
            user_messages = [user_messages[0]] + history + [user_messages[1]]

        if not chat:
            prompt = format_conversation(user_messages)
            prompt += "\nASSISTANT:<|START_RESPONSE|>"
            completion = self.answerer_model.client.completions.create(
                model=self.answerer_model.model_name,
                prompt=prompt,
                logprobs=5,
                # The line below reads index 0 only, so cap generation as the
                # chat twin above does. Every other completions call in this
                # file already caps at 6.
                max_tokens=6,
                temperature=0.0,
            )
            log_probs = completion.choices[0].logprobs.top_logprobs[0]
        else:
            log_probs = self._yes_no_probs_via_chat(
                model=self.guesser_model,
                messages=user_messages,
                top_logprobs=10,
                context="yes_no_probs",
            )

        return self._extract_yes_no_probs(log_probs)

    def answerer_log_probs(
        self,
        question: str,
        print_log_probs: bool = False,
        print_index_probs: bool = False,
    ):
        user_messages = self.prompt_builder.answerer_logprob_messages(
            self.item, question
        )
        gen = self.answerer_model.client.chat.completions.create(
            model=self.answerer_model.model_name,
            messages=user_messages,
            logprobs=True,
            top_logprobs=10,
            temperature=1.0,
        )
        response = gen.choices[0].message.content

        if self._is_answer_revealed(question):
            return {"role": "user", "content": "Bingo!"}
        return {"role": "user", "content": response}

    def answer_probs_given_history(self, question: str, chat: bool = True):
        user_messages = self.prompt_builder.history_answer_messages(
            self._get_guesser_messages().copy(), question
        )

        if not chat:
            prompt = format_conversation(user_messages)
            prompt += "\nASSISTANT:<|START_RESPONSE|>"
            response = self.answerer_model.client.completions.create(
                prompt=prompt,
                model=self.answerer_model.model_name,
                max_tokens=6,
                temperature=0.0,
                logprobs=5,
            )
            log_probs = response.choices[0].logprobs.top_logprobs[0]
        else:
            log_probs = self._yes_no_probs_via_chat(
                model=self.answerer_model,
                messages=user_messages,
                top_logprobs=5,
                context="answer_probs_given_history",
            )

        return self._extract_yes_no_probs(log_probs)

    def answerer(
        self, question: str, chat: bool = True, history=None, sample: bool = False
    ):
        user_messages = self.prompt_builder.answerer_messages(
            self.item, question, enable_maybe=self.enable_maybe
        )
        if history is not None:
            user_messages = [user_messages[0]] + history + [user_messages[1]]

        if not chat:
            prompt = format_conversation(user_messages)
            prompt += "\nASSISTANT:<|START_RESPONSE|>"
            completion = self.answerer_model.client.completions.create(
                prompt=prompt,
                model=self.answerer_model.model_name,
                max_tokens=6,
                temperature=0.0,
                logprobs=5,
            )
            response = self._clean_completion_response(completion.choices[0].text)
        else:

            def _call():
                completion = self.answerer_model.client.chat.completions.create(
                    model=self.answerer_model.model_name,
                    messages=user_messages,
                    stream=False,
                    **(
                        {"max_tokens": 10, "temperature": 1.0 if sample else 0.0}
                        if not self._is_reasoning_model(self.answerer_model.model_name)
                        else {}
                    ),
                )
                content = completion.choices[0].message.content
                if not content:
                    raise Exception("Empty response returned")
                return content

            response = self._retry(_call, context="answerer")

        if self._is_answer_revealed(question):
            return {"role": "user", "content": "Bingo!"}
        return {"role": "user", "content": response}

    def ask_llm(
        self, question: str, concept: str, chat: bool = True, seed: Optional[int] = None
    ):
        user_messages = self.prompt_builder.answerer_messages(concept, question)
        sampling = (
            {"temperature": 1.0, "seed": seed}
            if seed is not None
            else {"temperature": 0.0}
        )

        if not chat:
            prompt = format_conversation(user_messages)
            prompt += "\nASSISTANT:<|START_RESPONSE|>"
            response = self.answerer_model.client.completions.create(
                prompt=prompt,
                model=self.guesser_model.model_name,
                max_tokens=6,
                logprobs=5,
                **sampling,
            )
            return self._clean_completion_response(response.choices[0].text)

        def _call():
            response = self.guesser_model.client.chat.completions.create(
                model=self.guesser_model.model_name,
                messages=user_messages,
                **(
                    {"max_tokens": 6, **sampling}
                    if not self._is_reasoning_model(self.guesser_model.model_name)
                    else {}
                ),
            )
            return response.choices[0].message.content

        return self._retry(_call, context="ask_llm")

    def get_answer_probabilities(self):
        prompt = self.prompt_builder.answer_probability_prompt(
            self._get_guesser_messages(), prompt_suffix=self.prompt_suffix
        )
        theta_logprob = item_probs_from_completion_logprobs(
            prompt=prompt,
            possible_responses=self.item.split("|"),
            model=self.guesser_model,
            logprobs=True,
        )
        return np.logaddexp.reduce(theta_logprob, axis=0)

    def select_question_no_beliefs(self, followup_questions):
        """Pick a candidate question from the list alone (``Implicit_Max_no_beliefs``).

        The prompt carries the numbered candidates and nothing else — no
        transcript, no hypotheses — and the reply must match a candidate
        exactly, or candidate 1 is used. Contrast
        :meth:`select_question_with_beliefs`, which adds both and matches fuzzily.
        """
        prompt = "You are playing a 20 Questions game. You have the following candidate questions to ask:\n"
        for i, q in enumerate(followup_questions):
            prompt += f"{i + 1}. {q}\n"
        prompt += "Which question would you ask to best narrow down the possibilities? Please respond with the question only."

        def _call():
            response = self.guesser_model.client.chat.completions.create(
                model=self.guesser_model.model_name,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=50,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content.strip()
            for q in followup_questions:
                if response_text.lower() == q.lower():
                    return q
            return followup_questions[0]

        return self._retry(_call, context="select_question_no_beliefs")

    def select_question_with_beliefs(
        self,
        candidate_questions,
        history=None,
        temperature: float = 0.7,
        explain: bool = False,
        concepts=None,
    ):
        """Pick a candidate question given the transcript and the hypotheses (``Implicit_Max``).

        This is the LLM-judgment stand-in for the EIG argmax, so it sees what the
        EIG chooser sees: the conversation so far and the surviving hypotheses.
        The reply is matched back to a candidate exactly, then fuzzily above
        ``fuzzy_match_threshold``.
        """
        if not candidate_questions:
            raise ValueError("No candidate questions provided")

        messages_history = (
            history if history is not None else self._get_guesser_messages()
        )
        conversation_context = ""
        if messages_history and len(messages_history) > 1:
            q_count = 0
            for i in range(1, len(messages_history), 2):
                if i + 1 < len(messages_history):
                    q_count += 1
                    question = messages_history[i]["content"].strip()
                    answer = messages_history[i + 1]["content"].strip()
                    conversation_context += (
                        f"Question {q_count}: {question}\nAnswer: {answer}\n\n"
                    )

        system_prompt = "You are playing a game of 20 questions, trying to identify an unknown entity with yes/no questions.\n\n"
        if conversation_context:
            system_prompt += (
                f"Here is the conversation so far:\n{conversation_context}\n"
            )

        system_prompt += "Now you need to choose the best next question from the following candidates:\n"
        for i, q in enumerate(candidate_questions):
            system_prompt += f"{i + 1}. {q}\n"
        if explain:
            system_prompt += "\nPlease analyze which question would be most informative and provide the best information gain to narrow down the possibilities. First explain your reasoning, then provide your chosen question in the format: SELECTED QUESTION: <your chosen question>"
        if concepts:
            system_prompt += (
                f"\nThe current possible entities are: {', '.join(concepts)}\n"
            )
        system_prompt += "\nPlease select the best question that would most effectively narrow down the possibilities. Respond with only the exact text of the question you select."

        def _call():
            response = self.guesser_model.client.chat.completions.create(
                model=self.guesser_model.model_name,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=300 if explain else 50,
                temperature=temperature,
            )
            response_text = response.choices[0].message.content.strip()
            selected_question = response_text
            explanation = None

            if explain:
                match = re.search(
                    r"SELECTED QUESTION:\s*(.*?)(?:$|\n)", response_text, re.IGNORECASE
                )
                if match:
                    selected_question = match.group(1).strip()
                    explanation = response_text
                else:
                    lines = response_text.split("\n")
                    selected_question = lines[-1].strip()
                    explanation = "\n".join(lines[:-1])

            best_match = None
            best_score = 0
            for candidate in candidate_questions:
                if selected_question.lower() == candidate.lower():
                    return (candidate, explanation) if explain else candidate

                similarity = SequenceMatcher(
                    None, selected_question.lower(), candidate.lower()
                ).ratio()
                if similarity > best_score and similarity > self.fuzzy_match_threshold:
                    best_score = similarity
                    best_match = candidate

            if best_match:
                return (best_match, explanation) if explain else best_match

            print(
                f"No matching question found. Using first candidate. LLM response was: {selected_question}"
            )
            return (
                (candidate_questions[0], explanation)
                if explain
                else candidate_questions[0]
            )

        return self._retry(_call, context="select_question_with_beliefs")
