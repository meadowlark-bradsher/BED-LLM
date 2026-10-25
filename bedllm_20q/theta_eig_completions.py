from transformers import AutoTokenizer
import numpy as np
from functools import lru_cache
from serve_model import ModelServer, is_hosted_openai_model
from concurrent.futures import ThreadPoolExecutor


@lru_cache(maxsize=None)
def _tokenizer_for(model_name: str):
    """Return the tokenizer for ``model_name``.

    Cached because the tokenizer depends only on the model name and is not
    mutated here, while `item_probs_from_completion_logprobs` is called
    once per candidate question per answer branch on each turn.
    """
    if is_hosted_openai_model(model_name):
        # Hosted OpenAI models never reach this path: every caller routes them
        # through ``theta_eig.item_probs_from_chat_logprobs`` instead, and the
        # ``echo`` + ``logprobs`` completions call below is unavailable there.
        raise ValueError(
            f"{model_name!r} is a hosted OpenAI model; completion logprobs need "
            "a locally served model"
        )
    return AutoTokenizer.from_pretrained(model_name)


def item_probs_from_completion_logprobs(
    prompt: str,
    possible_responses: list[str],
    model: ModelServer,
    logprobs: bool = False,
):
    tokenizer = _tokenizer_for(model.model_name)
    client = model.client
    init_prompt = prompt
    init_prompt_len = len(tokenizer.encode(init_prompt, add_special_tokens=False))
    log_probabilities = np.zeros(len(possible_responses))

    def process_response(idx_response):
        idx, possible_response = idx_response
        pr_prompt = init_prompt + possible_response
        try:
            response = client.completions.create(
                model=model.model_name,
                prompt=pr_prompt,
                max_tokens=1,
                temperature=0.0,
                echo=True,
                logprobs=10,
            )

            tokens_response = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(possible_response)
            )
            overall_logprob = 0.0

            for i in range(len(tokens_response)):
                if (
                    tokens_response[i]
                    in response.choices[0].logprobs.top_logprobs[init_prompt_len + i]
                ):
                    prob = response.choices[0].logprobs.top_logprobs[
                        init_prompt_len + i
                    ][tokens_response[i]]
                    gprob = 0.0
                    if (
                        "Ġ" + tokens_response[i]
                        in response.choices[0].logprobs.top_logprobs[
                            init_prompt_len + i
                        ]
                    ):
                        gprob = response.choices[0].logprobs.top_logprobs[
                            init_prompt_len + i
                        ]["Ġ" + tokens_response[i]]
                        prob = np.logaddexp(prob, gprob)
                    overall_logprob += prob

            return idx, overall_logprob
        except Exception as e:
            print(f"Error processing '{possible_response}': {e}")
            return idx, float("-inf")

    with ThreadPoolExecutor(max_workers=min(len(possible_responses), 30)) as executor:
        results = list(executor.map(process_response, enumerate(possible_responses)))

    results.sort(key=lambda x: x[0])
    log_probabilities = [logprob for _, logprob in results]

    if logprobs:
        return log_probabilities
    else:
        return [np.exp(logprob) for logprob in log_probabilities]


if __name__ == "__main__":
    example_prompt = "Randomly sample one of the two co-founders of Google. Just provide the name without any additional information."
    example_prompt += "\nASSISTANT:<|START_RESPONSE|>"
    possible_responses = [
        "Sundar Pichai",
        "Larry Page",
        "Sergey Brin",
        "Fyodor Dostoyevsky",
    ]
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    model = ModelServer(model_name=model_name, port=8000, num_gpus=2)

    probabilities = item_probs_from_completion_logprobs(
        prompt=example_prompt, possible_responses=possible_responses, model=model
    )
    print("\nSummary of probabilities:")
    for i, (response, prob) in enumerate(zip(possible_responses, probabilities)):
        print(f"{i}. {response}: {prob:.10f}")
