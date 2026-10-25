import subprocess
import threading
import time
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI families that disallow `temperature` / `top_p` / `max_tokens`
# and use `max_completion_tokens` instead. Extend as new families ship.
REASONING_MODEL_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(name: str) -> bool:
    return any(name.startswith(p) for p in REASONING_MODEL_PREFIXES)


def is_hosted_openai_model(name: str) -> bool:
    """True when ``name`` is served by the OpenAI API, not a local vLLM process.

    The one place this distinction is made. Every routing site (which client
    to build, chat versus completions logprobs, which tokenizer) reads it, so
    a new hosted family is added to ``REASONING_MODEL_PREFIXES`` or here and
    nowhere else.
    """
    return name.startswith("gpt") or is_reasoning_model(name)


def supports_token_logprobs(name: str) -> bool:
    """True when the model can return per-token log-probabilities.

    A locally served model always can. A hosted ``gpt-4``-class chat model
    can. The OpenAI reasoning families cannot: they reject ``logprobs`` and
    ``temperature`` outright, so every acquisition that integrates p(yes | q,
    hypothesis) or scores hypotheses by their log-probability has nothing to
    read from them.
    """
    return not is_reasoning_model(name)


def resolve_model(model, model_name, *, default, port=8000, num_gpus=1):
    """Return ``(model_server, model_name)`` from a model instance, a name, or a default.

    A ``str`` ``model`` is treated as ``model_name``; a model instance is reused
    (reading ``model.model_name``); otherwise a ``ModelServer`` is constructed
    from ``model_name``, falling back to ``default``.
    """
    if isinstance(model, str):
        model_name = model
        model = None
    if model is not None:
        return model, model.model_name
    model_name = model_name or default
    return ModelServer(model_name=model_name, port=port, num_gpus=num_gpus), model_name


class ModelServer:
    def __init__(self, model_name, port=8000, dtype="bfloat16", num_gpus=1):
        self.model_name = model_name
        if is_hosted_openai_model(self.model_name):
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            parts = model_name.split("/")
            if len(parts) < 2:
                raise ValueError('local model_name must be of the form "org/name"')
            self.model_type = parts[1].split("-")[0]
            self.port = port
            self.dtype = dtype
            self.base_url = f"http://localhost:{self.port}/v1"
            self.num_gpus = num_gpus
            self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            self.process = self.start_server()

    def log_subprocess_output(self, pipe):
        for line in iter(pipe.readline, ""):
            print(line, end="")
        pipe.close()
        print("Server log tracking thread stopped.")

    def start_server(self):
        process = subprocess.Popen(
            [
                "vllm",
                "serve",
                self.model_name,
                "--port",
                str(self.port),
                "--dtype",
                self.dtype,
                "--tensor-parallel-size",
                str(self.num_gpus),
                # "--pipeline-parallel-size",
                # str(self.num_gpus),
                "--trust-remote-code",
                "--tokenizer-mode",
                "mistral" if self.model_type == "Mistral" else "auto",
                "--config-format",
                "mistral" if self.model_type == "Mistral" else "auto",
                "--load-format",
                "mistral" if self.model_type == "Mistral" else "auto",
                "--enforce-eager",
                "--max-model-len",
                "10000",
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                "1024",
                # "--max-seq-len-to-capture",
                # "2048",
                "--gpu-memory-utilization",
                "0.95",
            ]
            + (
                ["--enable-reasoning", "--reasoning-parser", "deepseek_r1"]
                if self.model_type == "DeepSeek" or self.model_type == "QwQ"
                else []
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_thread = threading.Thread(
            target=self.log_subprocess_output, args=(process.stdout,)
        )
        stderr_thread = threading.Thread(
            target=self.log_subprocess_output, args=(process.stderr,)
        )
        stdout_thread.start()
        stderr_thread.start()
        self._log_threads = (stdout_thread, stderr_thread)

        deadline = time.monotonic() + 1800
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {process.returncode} "
                    "before becoming ready"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("vLLM server did not become ready in time")
            try:
                response = requests.get(f"{self.base_url}/models", timeout=5)
                print(response.json())
                break
            except requests.exceptions.RequestException:
                time.sleep(1)

        return process

    def wait_termination(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            try:
                self.process.wait(timeout=15)
                print("Process terminated")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                print("Process killed.")
            for thread in getattr(self, "_log_threads", ()):
                thread.join(timeout=5)

        return wrapper

    @wait_termination
    def terminate(self):
        self.process.terminate()
