"""Shared test setup.

``pyproject.toml`` puts ``bedllm_20q/`` on ``sys.path``, because the package
uses flat imports and is run from inside that directory.

The model-serving libraries are heavy and none of the logic under test needs
them, so any that is not installed is replaced with a minimal stand-in. With
the full requirements installed the real modules are used. Tests that depend
on a library's behaviour monkeypatch the exact attribute they need, so they
pass either way.
"""

import importlib
import os
import sys
import types
from types import SimpleNamespace as NS

import pytest


def _stub_missing(name: str, **attrs) -> None:
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class _RequestException(Exception):
    pass


_stub_missing(
    "wandb",
    run=None,
    log=lambda *a, **k: None,
    Histogram=lambda x: x,
    Table=lambda columns: None,
    init=lambda **k: None,
    define_metric=lambda *a, **k: None,
    config=NS(update=lambda *a, **k: None),
    finish=lambda: None,
)
_stub_missing("tiktoken", encoding_for_model=lambda m: None)
_stub_missing("transformers", AutoTokenizer=NS(from_pretrained=lambda m: None))
_stub_missing("openai", OpenAI=lambda **k: NS())
_stub_missing("dotenv", load_dotenv=lambda: None)
_stub_missing(
    "requests",
    get=lambda *a, **k: None,
    exceptions=NS(RequestException=_RequestException),
)

# serve_model builds a real OpenAI client for hosted model names, and the
# client refuses to construct without a key.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture
def fake_model():
    """A stand-in for ModelServer: a model name and an inert client."""

    def _make(name: str):
        return NS(model_name=name, client=NS())

    return _make


@pytest.fixture
def run_config():
    """Parse ``eig_game_play`` flags from argv-style strings into a RunConfig.

    absl flags are process-global, so they are reset before each parse and
    again on teardown.
    """
    import eig_game_play as egp

    def _make(*args: str):
        egp.FLAGS.unparse_flags()
        egp.FLAGS(["prog", *args])
        return egp.RunConfig.from_flags(egp.FLAGS)

    yield _make
    egp.FLAGS.unparse_flags()


@pytest.fixture
def chat_messages():
    """A transcript in the ``[head, q1, a1, q2, a2, ...]`` layout."""
    return [
        {"role": "user", "content": "Let's play 20 questions."},
        {"role": "assistant", "content": "Is it a living thing?"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "content": "Is it a plant?"},
        {"role": "user", "content": "No."},
    ]
