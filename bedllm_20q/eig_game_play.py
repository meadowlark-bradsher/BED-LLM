from absl import app
from absl import flags
from serve_model import ModelServer
from question_sampler import QuestionSampler, QuestionSamplerConfig
from concept_sampler import ConceptSampler, ConceptSamplerConfig
from Q20 import Q20
from q20_question_choosers import VALID_ACQUISITIONS
from plot_success_rates import plot_success_rates
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import wandb
import traceback
import time
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("wandb").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# Number of consecutive batch iterations with no progress (no item completed)
# before the `while pending:` loop gives up.
_MAX_STUCK_ITERATIONS = 3

_DICTS_DIR = Path("dicts")
_FIGURES_DIR = Path("figures")

_DATASET_PATHS: dict[str, Path] = {
    "celebrities": Path("data") / "celebrities_test.txt",
    "things": Path("data") / "things_test.txt",
    "animals": Path("data") / "animals.txt",
}

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_name", "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic", None
)
flags.DEFINE_string("answerer_model_name", None, "name of the answerer model")
flags.DEFINE_integer("port", 8000, "port")
flags.DEFINE_integer(
    "num_gpus", 1, "the number of GPUs over which the parallelize the LLM"
)
flags.DEFINE_integer("num_turns", 20, "the number of turns to play the game")
flags.DEFINE_enum(
    "dataset",
    "celebrities",
    list(_DATASET_PATHS.keys()),
    "the dataset to use for the game",
)
flags.DEFINE_enum(
    "acquisition",
    "EIG",
    VALID_ACQUISITIONS,
    "the method of selecting questions",
)
flags.DEFINE_boolean(
    "sample_answer",
    False,
    "whether to sample the answer from the answerer model or use greedy decoding",
)
flags.DEFINE_boolean(
    "use_logits_entropy",
    True,
    "whether to use logits entropy for the question selection",
)
flags.DEFINE_boolean(
    "enable_maybe", False, 'whether answering "maybe" is allowed in the game'
)
flags.DEFINE_boolean("history", False, "whether to use history when generating answers")
flags.DEFINE_list(
    "multi_acquisitions", None, "comma-separated list of acquisition methods to use"
)
flags.DEFINE_boolean("debug", False, "whether to run in testing mode with limited data")
flags.DEFINE_integer(
    "num_questions", 15, "the number of questions to sample from the question sampler"
)
flags.DEFINE_integer(
    "max_workers", 25, "ThreadPoolExecutor worker count for the per-item batch"
)
flags.DEFINE_float(
    "retry_sleep_seconds",
    10.0,
    "sleep between iterations of the `while pending:` retry loop",
)
flags.DEFINE_boolean(
    "log_correct_guess_prob",
    True,
    "compute and log -log p(true_item | history) each turn. "
    "Only takes effect for non-GPT guesser models (needs logprobs).",
)

# ConceptSampler hyperparameters
flags.DEFINE_integer(
    "num_concept_samples", 15, "number of concept samples to generate per turn"
)
flags.DEFINE_float(
    "concept_temperature", 0.7, "sampling temperature for concept generation"
)
flags.DEFINE_integer(
    "n_completions", 3, "number of parallel completions for concept diversity"
)
flags.DEFINE_integer(
    "stage1_oversample_factor", 3, "oversampling multiplier at stage 1"
)
flags.DEFINE_integer(
    "concept_max_iterations", 2, "max retry iterations for concept generation"
)
flags.DEFINE_integer("tokens_per_sample", 15, "max_tokens budget per concept sample")
flags.DEFINE_integer(
    "multi_completion_stage_cutoff",
    19,
    "stage at which to fall back from N completions to 1 (endgame switch)",
)
flags.DEFINE_integer(
    "filter_oversample_factor",
    3,
    "multiplier applied to num_samples when capping the pre-filter sample pool",
)
flags.DEFINE_boolean(
    "use_logprob_filter",
    False,
    "filter hypotheses on the answer-token logprobs instead of on the text "
    "yes/no reply: reject a hypothesis when the model is confident that it "
    "contradicts the observed answer. Reads the same prompt, model and "
    "probabilities as the EIG, except that --history conditions the EIG prompt "
    "only. Falls back to the text check when the backend returns no usable "
    "logprobs",
)
flags.DEFINE_float(
    "logprob_reject_threshold",
    0.3,
    "reject a hypothesis when p(contradicting answer | question, hypothesis) is "
    "ABOVE this value, i.e. when the model gives the answer opposite to the "
    "observed one more than this much of the mass. 0.5 matches the text yes/no "
    "check; a lower value rejects more hypotheses, a higher value rejects fewer. "
    "Needs --use_logprob_filter",
)
flags.DEFINE_integer(
    "chat_retry_max_attempts", 10, "max retry attempts for chat completions"
)
flags.DEFINE_float(
    "chat_retry_initial_delay", 1.0, "initial backoff (seconds) for chat retries"
)
flags.DEFINE_float(
    "chat_retry_max_delay", 128.0, "max backoff (seconds) for chat retries"
)
flags.DEFINE_float(
    "chat_retry_backoff_multiplier", 2.0, "backoff multiplier for chat retries"
)
flags.DEFINE_integer(
    "chat_answer_max_tokens", 100, "max_tokens for the chat yes/no answer call"
)

# QuestionSampler hyperparameters
flags.DEFINE_float(
    "question_temperature", 0.8, "sampling temperature for question generation"
)
flags.DEFINE_float(
    "question_top_p", 0.9, "top-p (nucleus sampling) for question generation"
)
flags.DEFINE_float(
    "question_repetition_penalty",
    1.05,
    "repetition penalty for question generation (completions)",
)
flags.DEFINE_integer(
    "question_max_attempts", 2, "max outer retry attempts for question generation"
)


def _validate_acquisitions(flag_values):
    if flag_values is None:
        return True
    for method in flag_values:
        if method not in VALID_ACQUISITIONS:
            logger.error(
                "Invalid acquisition method: %s. Valid options are: %s",
                method,
                ", ".join(VALID_ACQUISITIONS),
            )
            return False
    return True


flags.register_validator(
    "multi_acquisitions",
    _validate_acquisitions,
    "must be a list of valid acquisition methods",
)

flags.register_validator(
    "logprob_reject_threshold",
    lambda value: 0.0 <= value <= 1.0,
    "must be a probability in [0, 1]",
)



@dataclass(frozen=True)
class RunConfig:
    """Resolved per-run configuration. One object, one place to read from."""

    model_name: str
    answerer_model_name: str
    port: int
    num_gpus: int
    num_turns: int
    dataset: str
    acquisitions: tuple[str, ...]
    sample_answer: bool
    use_logits_entropy: bool
    enable_maybe: bool
    history: bool
    debug: bool
    num_questions: int
    max_workers: int
    retry_sleep_seconds: float
    log_correct_guess_prob: bool
    concept_config: ConceptSamplerConfig
    question_config: QuestionSamplerConfig

    @classmethod
    def from_flags(cls, FLAGS) -> "RunConfig":
        answerer = FLAGS.answerer_model_name or FLAGS.model_name
        acquisitions = tuple(FLAGS.multi_acquisitions or [FLAGS.acquisition])
        concept_config = ConceptSamplerConfig(
            num_samples=FLAGS.num_concept_samples,
            temperature=FLAGS.concept_temperature,
            n_completions=FLAGS.n_completions,
            stage1_oversample_factor=FLAGS.stage1_oversample_factor,
            max_iterations=FLAGS.concept_max_iterations,
            tokens_per_sample=FLAGS.tokens_per_sample,
            multi_completion_stage_cutoff=FLAGS.multi_completion_stage_cutoff,
            filter_oversample_factor=FLAGS.filter_oversample_factor,
            chat_retry_max_attempts=FLAGS.chat_retry_max_attempts,
            chat_retry_initial_delay=FLAGS.chat_retry_initial_delay,
            chat_retry_max_delay=FLAGS.chat_retry_max_delay,
            chat_retry_backoff_multiplier=FLAGS.chat_retry_backoff_multiplier,
            chat_answer_max_tokens=FLAGS.chat_answer_max_tokens,
            use_logprob_filter=FLAGS.use_logprob_filter,
            logprob_reject_threshold=FLAGS.logprob_reject_threshold,
        )
        question_config = QuestionSamplerConfig(
            temperature=FLAGS.question_temperature,
            top_p=FLAGS.question_top_p,
            repetition_penalty=FLAGS.question_repetition_penalty,
            max_attempts=FLAGS.question_max_attempts,
        )
        return cls(
            model_name=FLAGS.model_name,
            answerer_model_name=answerer,
            port=FLAGS.port,
            num_gpus=FLAGS.num_gpus,
            num_turns=FLAGS.num_turns,
            dataset=FLAGS.dataset,
            acquisitions=acquisitions,
            sample_answer=FLAGS.sample_answer,
            use_logits_entropy=FLAGS.use_logits_entropy,
            enable_maybe=FLAGS.enable_maybe,
            history=FLAGS.history,
            debug=FLAGS.debug,
            num_questions=FLAGS.num_questions,
            max_workers=FLAGS.max_workers,
            retry_sleep_seconds=FLAGS.retry_sleep_seconds,
            log_correct_guess_prob=FLAGS.log_correct_guess_prob,
            concept_config=concept_config,
            question_config=question_config,
        )


@dataclass
class AcquisitionResult:
    """Output of `_run_acquisition`. Self-contained, serialisable, plottable."""

    acquisition: str
    items: list[str]
    successes: list[bool]
    cumulative_correct_by_turn: dict[int, int]
    theta_payload: dict
    # Number of items ever guessed correctly, derived from `first_correct_guess`
    # (same source of truth as `cumulative_correct_by_turn`). `successes` counts
    # the raw `game_play*` return values, which can use a different definition.
    correct_count: int


class ArtifactPaths:
    """Knows where every per-run artifact lives. Builds Paths, doesn't write."""

    def __init__(self, cfg: RunConfig, run_timestamp: str):
        self.cfg = cfg
        self.run_timestamp = run_timestamp

    def ensure_dirs(self) -> None:
        _DICTS_DIR.mkdir(parents=True, exist_ok=True)
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _filter_tag(self) -> str:
        """Return a filename segment naming the filter mode, or "" for the default.

        Only a logprob-filter run adds a segment, so two runs that differ just
        in the filter mode no longer write the same plot path.
        """
        concept_config = self.cfg.concept_config
        if not concept_config.use_logprob_filter:
            return ""
        return f"_lpfilter{concept_config.logprob_reject_threshold}"

    def _filename(
        self, prefix: str, acquisition: str, timestamp: str, suffix: str = ""
    ) -> Path:
        guesser_short = self.cfg.model_name.split("/")[-1]
        answerer_short = self.cfg.answerer_model_name.split("/")[-1]
        maybe_str = "maybe" if self.cfg.enable_maybe else ""
        return _DICTS_DIR / (
            f"{prefix}_{guesser_short}_{answerer_short}"
            f"_{self.cfg.dataset}_{acquisition}_{maybe_str}"
            f"{self._filter_tag()}_{timestamp}{suffix}.json"
        )

    def data(self, acquisition: str) -> Path:
        return self._filename("data", acquisition, self.run_timestamp)

    def theta(self, acquisition: str) -> Path:
        return self._filename("theta_probs", acquisition, self.run_timestamp)

    def partial_data(self, acquisition: str, timestamp: str, suffix: str) -> Path:
        return self._filename("partial_data", acquisition, timestamp, suffix)

    def partial_remaining(self, acquisition: str, timestamp: str, suffix: str) -> Path:
        return self._filename("partial_remaining", acquisition, timestamp, suffix)

    def plot(self) -> Path:
        guesser_short = self.cfg.model_name.split("/")[-1]
        answerer_short = self.cfg.answerer_model_name.split("/")[-1]
        return _FIGURES_DIR / (
            f"success_rates_{guesser_short}_{answerer_short}"
            f"_{self.cfg.dataset}{self._filter_tag()}.pdf"
        )



def load_items(dataset: str, debug: bool) -> list[str]:
    with open(_DATASET_PATHS[dataset], "r") as f:
        items = [line.strip() for line in f]
    if debug:
        items = items[:1]
    return items



def _shared_kwargs(cfg: RunConfig) -> dict:
    return {"num_turns": cfg.num_turns, "sample": cfg.sample_answer}


def _prompt_only_kwargs(_qs, _cs, cfg: RunConfig, _acq: str) -> dict:
    # Prompt_Only asks the guesser for the next question directly, so it needs
    # neither a hypothesis sampler nor a candidate-question sampler.
    return {**_shared_kwargs(cfg), "concept_sampler": None}


def _llm_chat_kwargs(qs, _cs, cfg: RunConfig, acq: str) -> dict:
    return {
        **_shared_kwargs(cfg),
        "concept_sampler": None,
        "question_sampler": qs,
        "acquisition": acq,
        "num_questions": cfg.num_questions,
    }


def _theta_eig_kwargs(qs, cs, cfg: RunConfig, acq: str) -> dict:
    return {
        **_shared_kwargs(cfg),
        "question_sampler": qs,
        "concept_sampler": cs,
        "num_questions": cfg.num_questions,
        "acquisition": acq,
    }


def _no_filter_kwargs(qs, cs, cfg: RunConfig, _acq: str) -> dict:
    # No_filter is EIG with filtering off - folded here rather than as its own
    # acquisition name on Q20 to mirror the previous behaviour.
    return {
        **_shared_kwargs(cfg),
        "question_sampler": qs,
        "concept_sampler": cs,
        "num_questions": cfg.num_questions,
        "acquisition": "EIG",
        "filter": False,
    }


def _eig_logits_only_kwargs(qs, cs, cfg: RunConfig, acq: str) -> dict:
    return {
        **_shared_kwargs(cfg),
        "question_sampler": qs,
        "concept_sampler": cs,
        "num_questions": cfg.num_questions,
        "acquisition": acq,
        "use_logits_entropy": cfg.use_logits_entropy,
    }


def _eig_full_kwargs(qs, cs, cfg: RunConfig, acq: str) -> dict:
    # `use_history` (the `--history` flag) is forwarded for EIG, Entropy and
    # Implicit_Max but is only consumed by the EIG chooser
    # (`question_eig`); the Entropy and Implicit_Max choosers ignore
    # it in `Q20._build_question_chooser`. So the flag is effectively EIG-only,
    # and sharing this builder is behaviourally equivalent to routing Entropy
    # and Implicit_Max through `_eig_logits_only_kwargs`.
    return {
        **_eig_logits_only_kwargs(qs, cs, cfg, acq),
        "use_history": cfg.history,
    }


@dataclass(frozen=True)
class _Dispatch:
    """The Q20 method to call + a kwargs builder for one acquisition strategy."""

    method_name: str
    build_kwargs: Callable[[QuestionSampler, ConceptSampler, RunConfig, str], dict]


_DISPATCH: dict[str, _Dispatch] = {
    "Prompt_Only":                    _Dispatch("game_play",           _prompt_only_kwargs),
    "Implicit_Max_no_beliefs":        _Dispatch("game_play",           _llm_chat_kwargs),
    "React":                          _Dispatch("game_play",           _llm_chat_kwargs),
    "Data_estimation_rank_aligned":   _Dispatch("game_play_theta_eig", _theta_eig_kwargs),
    "Data_estimation_union_mixture":  _Dispatch("game_play_theta_eig", _theta_eig_kwargs),
    "No_filter":                      _Dispatch("game_play_with_eig",  _no_filter_kwargs),
    "Sample_EIG":                     _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_uniform":                  _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_uniform_original":         _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_uniform_base":             _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_logprobs_original":        _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_logprobs":                 _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "Split_logprobs_renormalized":    _Dispatch("game_play_with_eig",  _eig_logits_only_kwargs),
    "EIG":                            _Dispatch("game_play_with_eig",  _eig_full_kwargs),
    "Entropy":                        _Dispatch("game_play_with_eig",  _eig_full_kwargs),
    "Implicit_Max":                   _Dispatch("game_play_with_eig",  _eig_full_kwargs),
}

# Keep the dispatch table and the validated-list of acquisitions in lockstep.
# If you add a strategy, add it to both places (or you will see a clear error
# at module load instead of a runtime KeyError later).
assert set(_DISPATCH.keys()) == set(VALID_ACQUISITIONS), (
    f"_DISPATCH / VALID_ACQUISITIONS mismatch: "
    f"only in _DISPATCH={set(_DISPATCH.keys()) - set(VALID_ACQUISITIONS)}, "
    f"only in VALID_ACQUISITIONS={set(VALID_ACQUISITIONS) - set(_DISPATCH.keys())}"
)


def _build_dispatch(
    acquisition: str,
    question_sampler: QuestionSampler,
    concept_sampler: ConceptSampler,
    cfg: RunConfig,
) -> tuple[str, dict]:
    """Return (Q20-method-name, kwargs) for the given acquisition strategy.

    Lookup-only; the mapping itself lives in `_DISPATCH` so adding a strategy
    is one new row (and, when novel, one kwargs-builder helper).
    """
    try:
        rule = _DISPATCH[acquisition]
    except KeyError:
        raise ValueError(
            f"Unknown acquisition {acquisition!r}. "
            f"Valid: {tuple(_DISPATCH.keys())}"
        ) from None
    return rule.method_name, rule.build_kwargs(
        question_sampler, concept_sampler, cfg, acquisition
    )


def _safe_json_dump(filename: Path, data, indent: int = 4) -> None:
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=indent)
        logger.info("Saved to %s", filename)
    except IOError as e:
        logger.error("Error saving to %s: %s", filename, e)


def _log_wandb(payload: dict) -> None:
    """`wandb.log` that no-ops when no wandb run is active (e.g. `--debug`)."""
    if wandb.run is not None:
        wandb.log(payload)


def _completed_first_guesses(agents_completed) -> list:
    """Turn numbers at which each completed agent first guessed correctly."""
    return [
        agent.first_correct_guess
        for agent in agents_completed.values()
        if getattr(agent, "first_correct_guess", None) is not None
    ]


def _cumulative_correct_by_turn(first_guesses, num_turns: int) -> dict:
    """Cumulative count of correct guesses at or before each turn."""
    return {
        turn: sum(1 for g in first_guesses if g <= turn)
        for turn in range(1, num_turns + 1)
    }


def _collect_theta_probs(agents_completed) -> dict:
    concept_probs_by_item = {}
    history_probs_by_item = {}
    for agent in agents_completed.values():
        item_key = getattr(agent, "item", f"item_{id(agent)}")
        concept_probs = getattr(agent, "turn_by_turn_theta_concept_probs", None)
        history_probs = getattr(agent, "turn_by_turn_theta_history_probs", None)
        if concept_probs:
            concept_probs_by_item[item_key] = concept_probs
        if history_probs:
            history_probs_by_item[item_key] = history_probs
    payload = {}
    if concept_probs_by_item:
        payload["concept_probs"] = concept_probs_by_item
    if history_probs_by_item:
        payload["history_probs"] = history_probs_by_item
    return payload


def _save_partial(
    paths: ArtifactPaths,
    acquisition: str,
    agents_completed,
    pending,
    batch_started_count: int,
    num_turns: int,
) -> None:
    completed_count = batch_started_count - len(pending)
    remaining_count = len(pending)
    first_correct_guesses = _completed_first_guesses(agents_completed)
    partial_cumulative = _cumulative_correct_by_turn(first_correct_guesses, num_turns)
    # Microsecond resolution so partial dumps from fast or back-to-back
    # iterations (and across acquisitions) cannot clobber each other.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_c{completed_count}_r{remaining_count}"

    _safe_json_dump(paths.partial_data(acquisition, ts, suffix), partial_cumulative)
    _safe_json_dump(
        paths.partial_remaining(acquisition, ts, suffix),
        sorted(list(pending)),
        indent=2,
    )


def _setup_models(cfg: RunConfig) -> tuple[ModelServer, ModelServer]:
    shared = ModelServer(
        model_name=cfg.model_name, port=cfg.port, num_gpus=cfg.num_gpus
    )
    if cfg.answerer_model_name != cfg.model_name:
        answerer = ModelServer(
            model_name=cfg.answerer_model_name, port=cfg.port, num_gpus=cfg.num_gpus
        )
    else:
        answerer = shared
    return shared, answerer


def _setup_samplers(
    model: ModelServer, cfg: RunConfig
) -> tuple[QuestionSampler, ConceptSampler]:
    qs = QuestionSampler(model=model, config=cfg.question_config)
    cs = ConceptSampler(model=model, config=cfg.concept_config)
    return qs, cs


def _setup_wandb(cfg: RunConfig) -> None:
    """Initialise wandb idempotently. No-op under --debug."""
    if cfg.debug or wandb.run is not None:
        return
    wandb.init(project="llm_q", group=cfg.model_name.split("/")[-1])
    wandb.define_metric("custom_step")
    wandb.define_metric("average_entropy/*", step_metric="custom_step")
    wandb.define_metric("ranking/*", step_metric="custom_step")
    wandb.config.update(asdict(cfg), allow_val_change=True)


def _run_acquisition(
    cfg: RunConfig,
    acquisition: str,
    items: list[str],
    *,
    guesser: ModelServer,
    answerer: ModelServer,
    qs: QuestionSampler,
    cs: ConceptSampler,
    paths: ArtifactPaths,
) -> AcquisitionResult:
    results: list[bool] = []
    pending = set(items)
    agents_completed: dict[str, Q20] = {}
    # Most recent failure per item that never completed, so a systematically
    # failing item is visible in the summary instead of silently lost.
    failed_items: dict[str, str] = {}
    stuck_iterations = 0
    previous_pending_size = len(pending)

    while pending:
        try:
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
                batch_started_count = len(pending)
                method_name, dispatch_kwargs = _build_dispatch(
                    acquisition, qs, cs, cfg
                )
                agents_by_item = {
                    item: Q20(
                        guesser_model=guesser,
                        answerer_model=answerer,
                        item=item,
                        task=cfg.dataset,
                        enable_maybe=cfg.enable_maybe,
                        react=(acquisition == "React"),
                        log_correct_guess_prob=cfg.log_correct_guess_prob,
                    )
                    for item in pending
                }
                future_to_item = {
                    executor.submit(
                        getattr(agent, method_name), **dispatch_kwargs
                    ): item
                    for item, agent in agents_by_item.items()
                }

                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                        agents_completed[item] = agents_by_item[item]
                        pending.remove(item)
                        failed_items.pop(item, None)
                        logger.info("Processed %s: %s", item, result)
                    except Exception as e:
                        failed_items[item] = repr(e)
                        logger.error("Error processing %s: %s", item, e)
                        traceback.print_exc()

                logger.info("Pending items: %s", pending)
                if pending:
                    _save_partial(
                        paths,
                        acquisition,
                        agents_completed,
                        pending,
                        batch_started_count,
                        cfg.num_turns,
                    )
        except Exception as e:
            logger.error("An error occurred: %s", e)
            traceback.print_exc()
            logger.info("Successes so far: %s", sum(results))

        # Stuck-detection runs every iteration regardless of how the batch
        # ended (clean exit or exception above), so the loop always either
        # makes progress or aborts.
        if pending:
            if len(pending) == previous_pending_size:
                stuck_iterations += 1
            else:
                stuck_iterations = 0
            previous_pending_size = len(pending)
            if stuck_iterations >= _MAX_STUCK_ITERATIONS:
                logger.error(
                    "Aborting acquisition %s: no progress after %d "
                    "iterations (still %d items pending: %s)",
                    acquisition,
                    stuck_iterations,
                    len(pending),
                    pending,
                )
                break
            time.sleep(cfg.retry_sleep_seconds)

    unfinished = {item: failed_items.get(item, "no result") for item in pending}
    if unfinished:
        logger.warning(
            "Acquisition %s: %d item(s) never completed: %s",
            acquisition,
            len(unfinished),
            unfinished,
        )

    first_correct_guesses = _completed_first_guesses(agents_completed)
    cumulative_correct_by_turn = _cumulative_correct_by_turn(
        first_correct_guesses, cfg.num_turns
    )
    for turn, count in cumulative_correct_by_turn.items():
        _log_wandb(
            {
                "correct_guesses/cumulative_correct_guesses": count,
                "custom_step": turn,
            }
        )
    logger.info(
        "Cumulative correct guesses by turn for %s: %s",
        acquisition,
        cumulative_correct_by_turn,
    )

    return AcquisitionResult(
        acquisition=acquisition,
        items=items,
        successes=results,
        cumulative_correct_by_turn=cumulative_correct_by_turn,
        theta_payload=_collect_theta_probs(agents_completed),
        correct_count=len(first_correct_guesses),
    )


def _save_result(result: AcquisitionResult, paths: ArtifactPaths) -> None:
    _safe_json_dump(paths.data(result.acquisition), result.cumulative_correct_by_turn)
    if result.theta_payload:
        _safe_json_dump(paths.theta(result.acquisition), result.theta_payload)
    logger.info(
        "Acquisition %s — correct=%d/%d",
        result.acquisition,
        result.correct_count,
        len(result.items),
    )


def _make_plots(
    results: list[AcquisitionResult], paths: ArtifactPaths, cfg: RunConfig
) -> None:
    datasets_with_names = [
        (result.cumulative_correct_by_turn, result.acquisition) for result in results
    ]
    guesser_short = cfg.model_name.split("/")[-1]
    plot_success_rates(
        data_with_names=datasets_with_names,
        plot_title=f"Success Rates for {guesser_short} on {cfg.dataset} Dataset",
        x_label="Turn Number",
        y_label="Success Rate (%)",
        include_error_bands=True,
        filename=str(paths.plot()),
    )


def main(_argv):
    cfg = RunConfig.from_flags(FLAGS)
    paths = ArtifactPaths(cfg, run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    paths.ensure_dirs()

    logger.info("dataset=%s", cfg.dataset)
    logger.info("acquisitions=%s", cfg.acquisitions)

    _setup_wandb(cfg)
    guesser, answerer = _setup_models(cfg)
    qs, cs = _setup_samplers(guesser, cfg)
    items = load_items(cfg.dataset, cfg.debug)
    logger.info("items=%s", items)

    start_time = time.time()
    results: list[AcquisitionResult] = []
    for acquisition in cfg.acquisitions:
        result = _run_acquisition(
            cfg,
            acquisition,
            items,
            guesser=guesser,
            answerer=answerer,
            qs=qs,
            cs=cs,
            paths=paths,
        )
        _save_result(result, paths)
        results.append(result)

    _make_plots(results, paths, cfg)
    logger.info("Elapsed time: %.2f seconds", time.time() - start_time)
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
