# BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design

Code for the 20 Questions experiments.

## Installation
```
pip install -r requirements.txt
```
To serve local models on GPUs, install the vLLM extras as well:
```
pip install -r requirements-local.txt
```

To run OpenAI models, please create a `.env` file in the root of the repository and set `OPENAI_API_KEY=your_openai_key`

## 20 Questions Experiments
To run 20 Questions Experiments, please `cd bedllm_20q` and run the following command
```
python eig_game_play.py --model_name {questioner model name} --answerer_model_name {answerer model name} --dataset {dataset to use}
```
The dataset can be one of `celebrities`, `animals` or `things`.
For example, to run the experiment on the `animals` dataset with `gpt-4.1` as the questioner and answerer we would run
```
python eig_game_play.py --model_name "gpt-4.1" --answerer_model_name "gpt-4.1" --dataset "animals"
```
By default the script runs one acquisition function, `EIG`. Pass
`--multi_acquisitions` to run several in one invocation, for example
`--multi_acquisitions EIG,Entropy,Prompt_Only`.

The results of the experiment will be saved as one JSON file per acquisition method in `bedllm_20q/dicts`, and the plot of the success rate vs. number of turns will appear as a pdf in `bedllm_20q/figures`.

### Running Local Models
The experiments are also configured to run local models on GPUs using `vllm`. To do this, you will need to specify the model name as above when running with 2 extra parameters: `port` and `num_gpus`. For example
```
python eig_game_play.py --model_name "Qwen/Qwen2.5-72B-Instruct" --answerer_model_name "gpt-4.1" --dataset "animals" --num_gpus 2 --port 8000
```

When the answerer is a *different* local model from the guesser, each gets its
own vLLM server. The answerer's server listens on `--answerer_port`, which
defaults to `--port + 1`. The launcher refuses to start if both are set to the
same port, and each server checks at startup that the model answering on its
port is the one it launched, so a leftover server from an earlier run is
reported rather than silently reused. Note that both servers are started with
the same `--num_gpus` and no GPU partitioning, and each claims 95% of the
memory on the GPUs it sees, so two local models need enough GPUs for both.

## Filtering hypotheses on the answer probability

Each turn, the code removes the sampled hypotheses that disagree with the
questions and answers so far. By default it reads the model's yes/no reply as
text and keeps a hypothesis only when that reply matches the observed answer.

`--use_logprob_filter` replaces the text comparison with a threshold on a
probability. For each hypothesis the code reads the log-probabilities of the
answer token and renormalises them over yes and no. The probability of the answer
*opposite* to the observed one is how confident the model is that the hypothesis
is wrong. The code rejects the hypothesis when that confidence goes above
`--logprob_reject_threshold` (default `0.3`):

```
python eig_game_play.py --model_name "gpt-4.1" --answerer_model_name "gpt-4.1" \
  --dataset "animals" --use_logprob_filter --logprob_reject_threshold 0.3
```

This is the same probability that the EIG integrates, read with the same prompt
and the same model, so the filter and the acquisition agree on one quantity.

The threshold is a knob on how much confidence the filter demands before it drops
a hypothesis:

- `0.5` matches the default text check. It rejects a hypothesis when the model
  finds the other answer more likely.
- Below `0.5` the filter rejects more hypotheses. At the `0.3` default a
  hypothesis goes when the model puts more than 30% on the other answer, so it
  needs 70% agreement to survive.
- Above `0.5` the filter rejects fewer hypotheses and drops one only when the
  model is close to certain.

A tie at exactly the threshold keeps the hypothesis, because rejection needs a
confidence above the threshold.

Four limits to know:

- The option needs a backend that returns chat-completions log-probabilities. If
  a request for them fails, the filter uses the text check for that hypothesis.
  When the replies show that the backend cannot serve the option at all, the
  filter logs one warning and uses the text check for the rest of the run.
- A reasoning model writes its chain of thought first, so the answer token is out
  of reach and no reply carries yes/no mass. The filter reports this the same way
  and falls back to the text check.
- `--history` makes the EIG condition on the transcript. The filter always asks
  about one question and one hypothesis, as the text check does, so with
  `--history` the two read different prompts.
- The text check rejects a hypothesis when it cannot read the reply as yes or no,
  for example "It depends.". The threshold rule has no such outcome: it
  renormalises over yes and no, so it decides on the ratio alone.

The option is off by default, so a run without it reproduces the recorded
results.

## Tests
```
pip install -r requirements-test.txt
pytest
```
Run from the repository root. The tests cover the acquisition math, the
reply parsing, flag resolution and dispatch, the model-routing predicates,
and the vLLM readiness check, all without a model or network. Any of the
model-serving libraries that is not installed is replaced with a stand-in
by `tests/conftest.py`, so a lighter environment with only `absl-py`,
`numpy`, `matplotlib` and `pytest` runs them too.

## Reproducibility
Set `PYTHONHASHSEED=0` when you launch any experiment:
```
PYTHONHASHSEED=0 python eig_game_play.py --model_name "gpt-4.1" --answerer_model_name "gpt-4.1" --dataset "animals"
```

## Citation
Paper: [arXiv:2508.21184](https://arxiv.org/abs/2508.21184)

```bibtex
@inproceedings{
choudhury2026bedllm,
title={{BED}-{LLM}: Intelligent Information Gathering with {LLM}s and Bayesian Experimental Design},
author={Deepro Choudhury and Sinead Williamson and Adam Golinski and Ning Miao and Freddie Bickford Smith and Michael Kirchhof and Yizhe Zhang and Tom Rainforth},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=qyylZMLYT8}
}
```

## License
MIT. See [LICENSE](LICENSE).
