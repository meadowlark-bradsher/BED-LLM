import tiktoken
import numpy as np
from prompts import task_completion_prompt
from retry_utils import retry_with_backoff
from serve_model import is_hosted_openai_model, supports_token_logprobs



def chat_with_logprobs(prompt, model, messages=None):
    if messages is None:
        messages_to_send = [
            {
                "role": "system",
                "content": "You are a helpful assistant playing 20 questions who outputs full names of celebrities based on questions and answers so far, including repeating parts of the name given to you.",
            },
            {"role": "user", "content": prompt},
        ]
    else:
        messages_to_send = [
            {
                "role": "system",
                "content": "You are a helpful assistant who outputs full names of celebrities, including repeating parts given to you.",
            }
        ]
        messages_to_send.extend(messages)
        messages_to_send.append({"role": "user", "content": prompt})

    def _log(attempt, total, e):
        print(f"Error during API call (attempt {attempt}/{total}): {e}")

    return retry_with_backoff(
        lambda: model.client.chat.completions.create(
            model=model.model_name,
            messages=messages_to_send,
            logprobs=True,
            temperature=0.0,
            top_logprobs=20,
        ),
        max_attempts=10,
        initial_delay=1.0,
        on_error=_log,
    )


def item_probs_from_chat_logprobs(
    model, items, messages=None, task="celebrities"
):
    # Byte-identical to the prompt the acquisition functions in `utils` use, so
    # both paths condition the guesser on the same request.
    init_prompt = task_completion_prompt(task)
    if task == "animals":
        prompt = (
            "Return only the exact name of one animal consistent with the questions and answers above.\n\n"
            "To increase randomness:\n"
            "1. Internally brainstorm a pool of diverse and representative animals.\n"
            "2. Avoid defaulting to the most obvious animals.\n"
            "Output rules:\n"
            "- Provide the **full** name without any additional information, including the tokens given. The first tokens have been generated for you: "
        )
    elif task == "things":
        prompt = (
            "Return only the exact name of one entity consistent with the questions and answers above.\n\n"
            "To increase randomness:\n"
            "1. Internally brainstorm a pool of diverse and representative entities/objects.\n"
            "2. Avoid defaulting to the most obvious or common entities.\n\n"
            "Output rules:\n"
            "- Provide the **full** name without any additional information, including the tokens given. The first tokens have been generated for you: "
        )
    elif task == "celebrities":
        prompt = """Return only the full name of one randomly selected famous person (living or deceased) consistent with the questions and answers above.
To increase randomness:
1. Internally brainstorm a pool of diverse and representative individuals.
2. Avoid defaulting to the most globally ubiquitous celebrities or famous figures.
Output rules:
- Provide the **full** name without any additional information, including the tokens given. The first tokens have been generated for you: """

    # This chat-logprob path relies on tiktoken's decode_single_token_bytes, so
    # it serves hosted OpenAI chat models only. Local models are routed through
    # theta_eig_completions.item_probs_from_completion_logprobs by the caller,
    # and the reasoning families return no logprobs for either path.
    name = model.model_name
    if not is_hosted_openai_model(name) or not supports_token_logprobs(name):
        raise ValueError(
            f"item_probs_from_chat_logprobs needs a hosted OpenAI chat model that "
            f"returns token log-probabilities, not {name!r}"
        )
    enc = tiktoken.encoding_for_model(model.model_name)
    print("Messages in probability function:", messages)
    response = chat_with_logprobs(init_prompt, model=model, messages=messages)
    print(response.choices[0].message.content)
    print(items)
    item_probababilities = []
    for item in items:
        tokens = enc.encode(item)
        overall_logprob = 0.0
        current_text = ""
        success = True

        first_token = enc.decode_single_token_bytes(tokens[0]).decode("utf-8")
        first_logprobs = {
            tl.token: tl.logprob
            for tl in response.choices[0].logprobs.content[0].top_logprobs
        }

        if first_token not in first_logprobs:
            item_probababilities.append(0.0)
            print(
                f"First token '{first_token}' not found in logprobs for item '{item}'."
            )
            continue

        overall_logprob += first_logprobs[first_token]
        current_text += first_token

        for i in range(1, len(tokens)):
            current_prompt = prompt + current_text
            token_response = chat_with_logprobs(current_prompt, model=model, messages=messages)

            # Each chat_with_logprobs() is a fresh single-shot completion, so the next token
            # generated under teacher forcing is always at content[0].
            token_logprobs = {
                tl.token: tl.logprob
                for tl in token_response.choices[0].logprobs.content[0].top_logprobs
            }
            next_token = enc.decode_single_token_bytes(tokens[i]).decode("utf-8")

            if next_token not in token_logprobs:
                success = False
                break

            overall_logprob += token_logprobs[next_token]
            current_text += next_token
        item_prob = np.exp(overall_logprob) if success else 0.0
        print(f"Item: {item}, Overall prob: {item_prob}")
        item_probababilities.append(item_prob)

    return item_probababilities
