from typing import Any, Callable, Dict, Optional, Protocol, Sequence


class GuessEvaluator(Protocol):
    def is_correct_guess(self, expected_item: str, guess: str) -> bool: ...


class RunLogger(Protocol):
    def create_table(self, columns: Sequence[str]) -> Any: ...

    def log_event(
        self, payload: Dict[str, Any], step: Optional[int] = None
    ) -> None: ...


class FunctionGuessEvaluator:
    def __init__(self, verify_fn: Callable[[str, str], bool]):
        self._verify_fn = verify_fn

    def is_correct_guess(self, expected_item: str, guess: str) -> bool:
        return self._verify_fn(expected_item, guess)


class WandbRunLogger:
    def __init__(self, wandb_api: Any):
        self._wandb = wandb_api

    def create_table(self, columns: Sequence[str]) -> Any:
        return self._wandb.Table(columns=list(columns))

    def log_event(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        # ``wandb.log`` raises before ``wandb.init``, and ``--debug`` never
        # initialises a run, so drop the event when no run is active.
        if self._wandb.run is None:
            return
        if step is None:
            self._wandb.log(payload)
        else:
            self._wandb.log(payload, step=step)
