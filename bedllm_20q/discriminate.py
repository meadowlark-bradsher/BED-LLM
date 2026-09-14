from typing import List

import openai

from retry_utils import retry_with_backoff
from serve_model import is_reasoning_model


def generate_discriminating_question(
    names: List[str],
    client: openai.OpenAI,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    temperature: float = 0.0,
    n: int = 1,
    task: str = "celebrities",
) -> List[str]:
    """Ask the LLM for the yes/no question that splits ``names`` most evenly.

    ``task`` names the entity type in the prompt. Returns one question string
    per sampled completion (``n`` completions at ``temperature``).
    """
    system_message = {
        "role": "user",
        "content": (
            "You are a specialized “yes/no question generator” whose job is to select "
            f"the single best binary question that most evenly splits a given list of {task}"
            "into two groups (ideally as close to half-and-half as possible). The question must be phrased in reference to 'the person' or 'the animal' or 'the entity'."
        ),
    }
    user_message = {
        "role": "user",
        "content": (
            "Here is the list of names:\n\n"
            f"```python\nnames = {names!r}\n```\n\n"
            "Output exactly one yes/no question phrased naturally, targeting the attribute "
            "that yields the most even split. Do not output anything else—no explanations, "
            "no code, just the question."
        ),
    }

    response = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            **{"temperature": temperature, "max_tokens": 1000}
            if not is_reasoning_model(model)
            else {},
            n=n,
        ),
        max_attempts=100,
        initial_delay=2,
        max_delay=128.0,
    )

    questions = []
    for choice in response.choices:
        question = choice.message.content.strip()
        questions.append(question)
    return questions
