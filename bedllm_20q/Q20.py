import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from utils import (
    question_eig,
    question_ensemble_entropy,
    text_mentions_item,
    question_proportion_entropy,
    question_theta_eig_union_mixture,
    question_split_score,
    question_theta_eig_rank_aligned,
    question_eig_sampled,
)
from q20_dependencies import (
    GuessEvaluator,
    RunLogger,
    FunctionGuessEvaluator,
    WandbRunLogger,
)
from q20_question_choosers import (
    QuestionChooser,
    PartitionQuestionChooser,
    FunctionQuestionChooser,
    LlmQuestionChooser,
    FunctionPartitionQuestionChooser,
)
from q20_turn_setup import (
    TurnSetup,
    NaiveTurnSetup,
    EigTurnSetup,
    ThetaTurnSetup,
)
from q20_llm_io import Q20LlmInterface, DefaultQ20LlmInterface, yes_no_polarity
from prompts import task_completion_prompt
from serve_model import is_hosted_openai_model, is_reasoning_model
import wandb
from retry_utils import retry_with_backoff


@dataclass
class GameState:
    guesser_win: bool = False
    guesser_messages: List[Dict[str, str]] = field(default_factory=list)
    first_correct_guess: Optional[int] = None
    turn_by_turn_theta_history_probs: List[Any] = field(default_factory=list)
    turn_by_turn_theta_concept_probs: List[Any] = field(default_factory=list)


class Q20:
    MAX_RETRIES = 10
    EPSILON = 1e-6
    FUZZY_MATCH_THRESHOLD = 0.8

    def __init__(
        self,
        guesser_model,
        answerer_model,
        item,
        task=None,
        enable_maybe=False,
        react=False,
        log_correct_guess_prob: bool = True,
        guess_evaluator: Optional[GuessEvaluator] = None,
        run_logger: Optional[RunLogger] = None,
        llm_interface: Optional[Q20LlmInterface] = None,
    ):

        if react:
            self.first_user_utterance = (
                f"Your task is to ask a series of questions to deduce the identity of the "
                f"{'person' if task == 'celebrities' else 'animal' if task == 'animals' else 'entity'} "
                f"that I'm thinking of with as few queries as possible. "
                f"Only ask questions that can be answered by 'yes', 'no' or 'maybe'. "
                f"Do not ask for hints. The actual question you ask must be standalone with no linebreakers. "
                "Use the following format for every response:\n"
                "Thought: <brief reasoning about what to ask next>\n"
                'Action: ASK["<a single yes/no/maybe question with no linebreaks>"]\n\n'
                "Now start by producing your first Thought and Action."
            )
        else:
            self.first_user_utterance = (
                f"Your task is to ask a series of questions to deduce the identity of the {'person' if task == 'celebrities' else 'animal' if task == 'animals' else 'entity'} "
                f"that I'm thinking of with as few queries as possible. "
                f"Only ask questions that can be answered by 'yes', 'no' or 'maybe'. "
                f"Do not ask for hint. Make your question standalone with no linebreaker. "
                # f"Output format: <question>Your question here?</question>"
                f"Now start asking a question."
            )

        self.state = GameState()
        self.vicuna_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.guesser_model = guesser_model
        self.answerer_model = answerer_model
        self.item = item
        self.task = task
        self.enable_maybe = enable_maybe
        self.log_correct_guess_prob = log_correct_guess_prob
        self.guess_evaluator = (
            guess_evaluator
            if guess_evaluator is not None
            else FunctionGuessEvaluator(text_mentions_item)
        )
        self.run_logger = (
            run_logger if run_logger is not None else WandbRunLogger(wandb)
        )
        self.llm_interface = (
            llm_interface
            if llm_interface is not None
            else DefaultQ20LlmInterface(
                guesser_model=self.guesser_model,
                answerer_model=self.answerer_model,
                item=self.item,
                enable_maybe=self.enable_maybe,
                prompt_suffix=task_completion_prompt(task),
                get_guesser_messages=lambda: self.guesser_messages,
                retry=self._retry,
                is_reasoning_model=self._is_reasoning_model,
                is_answer_revealed=self._is_answer_revealed,
                epsilon=self.EPSILON,
                fuzzy_match_threshold=self.FUZZY_MATCH_THRESHOLD,
            )
        )

    @property
    def guesser_win(self):
        return self.state.guesser_win

    @guesser_win.setter
    def guesser_win(self, value):
        self.state.guesser_win = value

    @property
    def guesser_messages(self):
        return self.state.guesser_messages

    @guesser_messages.setter
    def guesser_messages(self, value):
        self.state.guesser_messages = value

    @property
    def first_correct_guess(self):
        return self.state.first_correct_guess

    @first_correct_guess.setter
    def first_correct_guess(self, value):
        self.state.first_correct_guess = value

    @property
    def turn_by_turn_theta_history_probs(self):
        return self.state.turn_by_turn_theta_history_probs

    @turn_by_turn_theta_history_probs.setter
    def turn_by_turn_theta_history_probs(self, value):
        self.state.turn_by_turn_theta_history_probs = value

    @property
    def turn_by_turn_theta_concept_probs(self):
        return self.state.turn_by_turn_theta_concept_probs

    @turn_by_turn_theta_concept_probs.setter
    def turn_by_turn_theta_concept_probs(self, value):
        self.state.turn_by_turn_theta_concept_probs = value


    def _retry(self, fn, context="api_call", max_retries=None):
        if max_retries is None:
            max_retries = self.MAX_RETRIES

        def _log(attempt, total, e):
            print(f"Error in {context} (attempt {attempt}/{total}): {e}")

        return retry_with_backoff(
            fn, max_attempts=max_retries, initial_delay=1.0, on_error=_log
        )

    @staticmethod
    def _clean_question_text(text):
        text = re.sub(r"the entity you are thinking of", "it", text)
        text = re.sub(r"the entity you're thinking of", "it", text)
        text = re.sub(r" you're thinking of", "", text)
        text = re.sub(r" you are thinking of", "", text)
        return text

    def _is_answer_revealed(self, question):
        return text_mentions_item(self.item, question)

    def _is_correct_guess(self, guess):
        return self.guess_evaluator.is_correct_guess(self.item, guess)

    def _create_table(self, columns):
        return self.run_logger.create_table(columns)

    def _log_event(self, payload, step=None):
        self.run_logger.log_event(payload, step=step)

    def _record_first_correct_guess(self, turn_number):
        if self.first_correct_guess is None:
            self.first_correct_guess = turn_number
            print(f"First correct guess for {self.item} at turn {turn_number}")

    def _resolve_guesser_message(self, turn_index, user_mode, generated_question=None):
        if (not user_mode) or user_mode is None:
            if generated_question is None:
                raise ValueError(f"Missing generated question at turn {turn_index + 1}")
            return {
                "role": "assistant",
                "content": self._clean_question_text(generated_question),
            }

        user_q = input(
            f"Type in your questions for turn {turn_index + 1}. (e.g. Is it a living thing?)\n"
        )
        return {"role": "assistant", "content": user_q}

    def _run_turn_exchange(
        self,
        guesser_msg,
        turn_index,
        num_turns,
        print_probs=False,
        sample=False,
        extra_question=None,
        question_override=None,
    ):
        self.guesser_messages.append(guesser_msg)
        # ``question_override`` splits what the transcript records from what the
        # answerer is asked. ReAct needs that split: its reply carries a
        # ``Thought:`` line that belongs in the transcript but must not be put
        # to the answerer, whose reveal check scans the question for the target.
        guesser_question = (
            question_override.strip()
            if question_override
            else guesser_msg["content"].strip()
        )
        print("Guesser: ", guesser_msg["content"])

        if turn_index == num_turns - 1:
            self.guesser_messages[-1]["content"] = (
                self.guesser_messages[-1]["content"] + " Is it right?"
            )

        if print_probs:
            usr_msg = self.answerer_log_probs(guesser_question)
        else:
            usr_msg = self.answerer(guesser_question, sample=sample)

        extra_usr_msg = None
        if extra_question is not None and not print_probs:
            extra_usr_msg = self.answerer(extra_question, chat=True)

        self.guesser_messages.append(
            {"role": "user", "content": f"{usr_msg['content'].strip()}"}
        )
        print("Answerer: ", usr_msg["content"].strip())

        if not self._is_api_model(self.guesser_model.model_name):
            overall_prob_theta = self.get_answer_probabilities()
            self.turn_by_turn_theta_history_probs.append(-overall_prob_theta)
            print("Overall probabilities of possible answers: ", overall_prob_theta)

        return usr_msg, extra_usr_msg

    @staticmethod
    def _is_bingo_response(usr_msg):
        return "bingo" in usr_msg["content"].lower()

    @staticmethod
    def _concepts_after_answer(answer, concepts, yes_concepts, no_concepts):
        """Return the hypothesis pool to carry forward after ``answer``.

        The theta-EIG chooser samples one pool per answer branch; the
        answer picks the branch. Only the leading yes/no token counts, so an
        uncertain reply such as "Don't know." (the ``--enable_maybe`` option)
        or "Unknown" keeps the current pool instead of being read as a No
        because it contains the letters "no".
        """
        polarity = yes_no_polarity(answer)
        if polarity is True:
            return yes_concepts
        if polarity is False:
            return no_concepts
        print("Uncertain answer, keeping all concepts.")
        return concepts

    def _set_guess_deadline_instruction(self, turn_index, num_turns):
        if turn_index == num_turns - 2:
            self.guesser_messages[-1]["content"] = (
                self.guesser_messages[-1]["content"] + " You must guess now, what's it?"
            )

    def _finalize_turn(self, usr_msg, turn_index, num_turns, success_payload):
        if self._is_bingo_response(usr_msg):
            self.guesser_win = True
            self._record_first_correct_guess(turn_index + 1)
            self._log_event(success_payload)
            return True

        self._set_guess_deadline_instruction(turn_index, num_turns)
        return False

    def _build_question_chooser(
        self, acquisition, use_logits_entropy=False, use_history=False
    ) -> QuestionChooser:
        if acquisition == "EIG":
            return FunctionQuestionChooser(
                question_eig, use_history=use_history
            )
        if acquisition == "Sample_EIG":
            return FunctionQuestionChooser(question_eig_sampled, K=5)
        if acquisition == "Entropy":
            if use_logits_entropy:
                return FunctionQuestionChooser(question_ensemble_entropy)
            return FunctionQuestionChooser(question_proportion_entropy)
        if acquisition == "Implicit_Max":
            return LlmQuestionChooser(self.select_question_with_beliefs)
        if acquisition == "Split_uniform":
            return FunctionQuestionChooser(
                question_split_score, original=False
            )
        if acquisition == "Split_uniform_original":
            return FunctionQuestionChooser(
                question_split_score, original=True
            )
        if acquisition == "Split_uniform_base":
            return FunctionQuestionChooser(
                question_split_score, original=True, base=True
            )
        if acquisition == "Split_logprobs_original":
            return FunctionQuestionChooser(
                question_split_score, original=True, logprobs=True
            )
        if acquisition == "Split_logprobs":
            return FunctionQuestionChooser(
                question_split_score, original=False, logprobs=True
            )
        if acquisition == "Split_logprobs_renormalized":
            return FunctionQuestionChooser(
                question_split_score,
                original=False,
                logprobs=True,
                renormalize=True,
            )
        raise ValueError(f"Unknown acquisition: {acquisition}")

    def _build_partition_question_chooser(
        self, acquisition
    ) -> PartitionQuestionChooser:
        # Both acquisitions run `game_play_theta_eig`; they differ only in how
        # the Yes and No branch posteriors are combined. See the two functions
        # in `utils` for the formulae.
        if acquisition == "Data_estimation_union_mixture":
            return FunctionPartitionQuestionChooser(question_theta_eig_union_mixture)
        if acquisition == "Data_estimation_rank_aligned":
            return FunctionPartitionQuestionChooser(question_theta_eig_rank_aligned)
        raise ValueError(f"Unknown theta acquisition: {acquisition}")

    def _build_naive_turn_setup(
        self, question_sampler, acquisition, num_questions
    ) -> TurnSetup:
        return NaiveTurnSetup(
            self,
            question_sampler=question_sampler,
            acquisition=acquisition,
            num_questions=num_questions,
        )

    def _build_eig_turn_setup(
        self, question_sampler, concept_sampler, num_questions, filter_value
    ) -> TurnSetup:
        return EigTurnSetup(
            self,
            question_sampler=question_sampler,
            concept_sampler=concept_sampler,
            num_questions=num_questions,
            filter_value=filter_value,
        )

    def _build_theta_turn_setup(self, question_sampler, num_questions) -> TurnSetup:
        return ThetaTurnSetup(
            self,
            question_sampler=question_sampler,
            num_questions=num_questions,
        )

    def _is_reasoning_model(self, model_name):
        return is_reasoning_model(model_name)

    @staticmethod
    def _is_api_model(model_name):
        return is_hosted_openai_model(model_name)


    def dialog_history(self):
        history = self.vicuna_prompt + " "
        for item in self.guesser_messages:
            if item["role"].upper() == "USER":
                history += "USER: " + item["content"]
            elif item["role"].upper() == "ASSISTANT":
                history += " " + "ASSISTANT: " + item["content"] + "</s>"
        return history

    def guesser_chat(self, messages=None, greedy_decode=False):
        return self.llm_interface.guesser_chat(
            messages=messages, greedy_decode=greedy_decode
        )

    def yes_no_probs(self, question, concept, chat=True, use_history=False):
        return self.llm_interface.yes_no_probs(
            question=question,
            concept=concept,
            chat=chat,
            use_history=use_history,
        )

    def answerer_log_probs(
        self, question, print_log_probs=False, print_index_probs=False
    ):
        return self.llm_interface.answerer_log_probs(
            question=question,
            print_log_probs=print_log_probs,
            print_index_probs=print_index_probs,
        )

    def answer_probs_given_history(self, question, chat=True):
        return self.llm_interface.answer_probs_given_history(
            question=question, chat=chat
        )

    def answerer(self, question, chat=True, history=None, sample=False):
        return self.llm_interface.answerer(
            question=question,
            chat=chat,
            history=history,
            sample=sample,
        )

    def ask_llm(self, question, concept, chat=True, seed=None):
        return self.llm_interface.ask_llm(
            question=question, concept=concept, chat=chat, seed=seed
        )

    def reset(self):
        self.guesser_messages = [
            {
                "role": "user",
                "content": self.first_user_utterance,
            }
        ]

    def get_answer_probabilities(self):
        return self.llm_interface.get_answer_probabilities()

    def game_play(
        self,
        user_mode=False,
        num_turns=20,
        print_probs=False,
        concept_sampler=None,
        sample=False,
        question_sampler=None,
        acquisition="Naive",
        num_questions=15,
    ):
        self.reset()
        print("dialogue history: ", self.dialog_history())

        columns = ["Question", "Answer", "EIG"]
        table = self._create_table(columns=columns)
        turn_setup = self._build_naive_turn_setup(
            question_sampler=question_sampler,
            acquisition=acquisition,
            num_questions=num_questions,
        )
        for t in range(num_turns):
            turn_setup_data = turn_setup.prepare(
                turn_index=t,
                num_turns=num_turns,
                user_mode=user_mode,
            )
            guesser_msg = self._resolve_guesser_message(
                t,
                user_mode=user_mode,
                generated_question=turn_setup_data.generated_question,
            )
            usr_msg, _ = self._run_turn_exchange(
                guesser_msg,
                turn_index=t,
                num_turns=num_turns,
                print_probs=print_probs,
                sample=sample,
                question_override=turn_setup_data.answerer_question,
            )

            table.add_data(guesser_msg["content"], usr_msg["content"], "N/A")

            if self._finalize_turn(
                usr_msg,
                turn_index=t,
                num_turns=num_turns,
                success_payload={f"Successful Naive/Q/A Table for {self.item}": table},
            ):
                return True

        self._log_event({f"Unsuccessful Naive/Q/A Table for {self.item}": table})
        return False

    def game_play_with_eig(
        self,
        question_sampler,
        concept_sampler,
        user_mode=False,
        num_turns=20,
        print_probs=False,
        acquisition="EIG",
        sample=False,
        use_logits_entropy=False,
        num_questions=10,
        filter=True,
        use_history=False,
    ):
        self.reset()
        print("dialogue history: ", self.dialog_history())
        columns = [
            "Question",
            "Answer",
            "Concepts",
            "Questions",
            "EIG",
            "Guess",
        ]
        q_table = self._create_table(columns=columns)
        turn_setup = self._build_eig_turn_setup(
            question_sampler=question_sampler,
            concept_sampler=concept_sampler,
            num_questions=num_questions,
            filter_value=filter,
        )
        # Stateless and built from loop-invariant inputs, so construct it once.
        chooser = self._build_question_chooser(
            acquisition,
            use_logits_entropy=use_logits_entropy,
            use_history=use_history,
        )
        for t in range(num_turns):
            # System asking a question
            print(f"Question {t + 1}")
            max_eig = 0
            generated_question = None
            turn_setup_data = turn_setup.prepare(
                turn_index=t,
                num_turns=num_turns,
                user_mode=user_mode,
            )
            concepts = list(turn_setup_data.concepts)
            followup_questions = list(turn_setup_data.followup_questions)
            guess = turn_setup_data.guess
            is_auto_mode = (not user_mode) or user_mode is None
            if is_auto_mode:
                if turn_setup_data.pre_guess_correct:
                    self.guesser_win = True
                    self._log_event(
                        {f"Successful {acquisition}/Q/A Table for {self.item}": q_table}
                    )
                    return True

                if acquisition == "Sample_EIG":
                    print("Using Sample_EIG acquisition")

                question, max_eig = chooser.choose(
                    concepts,
                    followup_questions,
                    agent=self,
                    item=self.item,
                    step=t + 1,
                )
                if acquisition == "Implicit_Max":
                    print("Question selected by LLM judgment: ", question)

                generated_question = question

            guesser_msg = self._resolve_guesser_message(
                t, user_mode=user_mode, generated_question=generated_question
            )
            usr_msg, _ = self._run_turn_exchange(
                guesser_msg,
                turn_index=t,
                num_turns=num_turns,
                print_probs=print_probs,
                sample=sample,
            )

            q_table.add_data(
                guesser_msg["content"],
                usr_msg["content"],
                str(concepts),
                str(followup_questions),
                max_eig,
                guess,
            )

            if self._finalize_turn(
                usr_msg,
                turn_index=t,
                num_turns=num_turns,
                success_payload={
                    f"Successful {acquisition}/Q/A Table for {self.item}": q_table
                },
            ):
                return True

        self._log_event(
            {f"Unsuccessful {acquisition}/Q/A Table for {self.item}": q_table}
        )
        return False

    def select_question_no_beliefs(self, followup_questions):
        return self.llm_interface.select_question_no_beliefs(
            followup_questions=followup_questions
        )

    def select_question_with_beliefs(
        self,
        candidate_questions,
        history=None,
        temperature=0.7,
        explain=False,
        concepts=None,
    ):
        return self.llm_interface.select_question_with_beliefs(
            candidate_questions=candidate_questions,
            history=history,
            temperature=temperature,
            explain=explain,
            concepts=concepts,
        )

    def game_play_theta_eig(
        self,
        question_sampler,
        concept_sampler,
        user_mode=False,
        num_turns=20,
        sample=False,
        num_questions=10,
        acquisition=None,
    ):
        self.reset()
        concepts = []
        columns = ["Question", "Answer", "Concepts", "Questions", "EIG", "Guess"]
        q_table = self._create_table(columns=columns)
        turn_setup = self._build_theta_turn_setup(
            question_sampler=question_sampler,
            num_questions=num_questions,
        )
        for t in range(num_turns):
            print(f"Question {t + 1}")
            max_eig = 0
            yes_concepts = concepts
            no_concepts = concepts
            turn_setup_data = turn_setup.prepare(
                turn_index=t,
                num_turns=num_turns,
                user_mode=user_mode,
                current_concepts=concepts,
            )
            followup_questions = list(turn_setup_data.followup_questions)
            guess = turn_setup_data.guess
            generated_question = None
            if (not user_mode) or user_mode is None:
                if turn_setup_data.pre_guess_correct:
                    self.guesser_win = True
                    self._log_event({f"Successful/Q/A Table for {self.item}": q_table})
                    return True

                chooser = self._build_partition_question_chooser(acquisition)
                question_eig, max_eig, yes_concepts, no_concepts = chooser.choose(
                    list(turn_setup_data.concepts),
                    followup_questions,
                    agent=self,
                    item=self.item,
                    step=t + 1,
                    concept_sampler=concept_sampler,
                )
                print("Yes concepts: ", yes_concepts)
                print("No concepts: ", no_concepts)
                generated_question = question_eig

            guesser_msg = self._resolve_guesser_message(
                t, user_mode=user_mode, generated_question=generated_question
            )
            usr_msg, _ = self._run_turn_exchange(
                guesser_msg,
                turn_index=t,
                num_turns=num_turns,
                sample=sample,
            )
            concepts = self._concepts_after_answer(
                usr_msg["content"], concepts, yes_concepts, no_concepts
            )

            q_table.add_data(
                guesser_msg["content"],
                usr_msg["content"],
                str(concepts),
                str(followup_questions),
                max_eig,
                guess,
            )

            if self._finalize_turn(
                usr_msg,
                turn_index=t,
                num_turns=num_turns,
                success_payload={f"Successful/Q/A Table for {self.item}": q_table},
            ):
                return True

        self._log_event({f"Unsuccessful/Q/A Table for {self.item}": q_table})
        return False
