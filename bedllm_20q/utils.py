import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import wandb
import re

from prompts import task_completion_prompt
from serve_model import is_hosted_openai_model, is_reasoning_model
from theta_eig import item_probs_from_chat_logprobs
from theta_eig_completions import (
    item_probs_from_completion_logprobs,
)

_ACQUISITION_MAX_WORKERS = 16


def _acquisition_max_workers(concepts) -> int:
    return max(1, min(len(concepts) + 1, _ACQUISITION_MAX_WORKERS))


def format_conversation(messages) -> str:
    """Serialize a chat-message list into a flat ``Role: content`` transcript.

    This is the join only: callers add any completion-prompt suffix (e.g.
    ``"\\nASSISTANT:<|START_RESPONSE|>"``) or task-prompt header themselves,
    since those conventions vary by call site.
    """
    return "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
    )


def _concept_probs_given_history(agent, messages, candidate_concepts):
    """Return the guesser's normalized probability over ``candidate_concepts``.

    Hosted OpenAI chat models are teacher-forced through
    :func:`theta_eig.item_probs_from_chat_logprobs`; locally served models
    are scored on a flat completion prompt via
    :func:`theta_eig_completions.item_probs_from_completion_logprobs`. The
    OpenAI reasoning families return no token log-probabilities at all, so
    they are rejected here rather than ten retries deep inside either path.

    ``question_split_score``, ``question_theta_eig_union_mixture`` and
    ``question_theta_eig_rank_aligned`` all use this quantity. The prompt wording
    is part of the conditioning distribution, so it is defined once here.
    """
    if len(candidate_concepts) == 0:
        return np.array([])
    task = getattr(agent, "task", "celebrities")
    model_name = str(getattr(agent.guesser_model, "model_name", ""))
    if is_reasoning_model(model_name):
        raise ValueError(
            f"{model_name!r} is an OpenAI reasoning model and returns no token "
            "log-probabilities; the logprob-based acquisitions need a gpt-4-class "
            "model or a locally served one as the guesser"
        )
    if is_hosted_openai_model(model_name):
        probs = item_probs_from_chat_logprobs(
            agent.guesser_model,
            candidate_concepts,
            messages=messages,
            task=task,
        )
    else:
        # Built inside the branch: the gpt path passes `messages` straight
        # through and never reads a flattened transcript.
        prompt = (
            "Here is the conversation history with the user:\n"
            + format_conversation(messages)
            + "\n"
            + task_completion_prompt(task)
            + "\nASSISTANT:<|START_RESPONSE|>"
        )
        probs = item_probs_from_completion_logprobs(
            prompt=prompt,
            possible_responses=candidate_concepts,
            model=agent.guesser_model,
        )
    probs = np.array(probs, dtype=float)
    probs /= np.sum(probs) + 1e-10
    return probs


def _safe_entropy_term(p):
    """Return one binary-entropy term in bits. Return 0.0 at the p in {0, 1} limits.

    The return type is always float, so a list of unanimous scores does not
    build an int64 array and write a different dtype to the run table.
    """
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p)


def ensemble_binary_entropy(probs_yes):
    """Return the Shannon entropy (bits) of the ensemble distribution.

    ``probs_yes`` has shape (K,) and gives p(yes | x, theta_i) for each of the K
    posterior samples; the entropy is taken over their average.
    """
    if len(probs_yes) == 0:
        return 0.0
    p_yes_avg = sum(probs_yes) / len(probs_yes)

    p_yes_avg = min(max(p_yes_avg, 0.0), 1.0)

    p_no_avg = 1.0 - p_yes_avg

    return _safe_entropy_term(p_yes_avg) + _safe_entropy_term(p_no_avg)


def question_proportion_entropy(
    concepts, followup_questions, agent, item=None, step=None
):
    if len(concepts) == 0:
        return followup_questions[0], 0.0
    ensemble_entropies = []
    with ThreadPoolExecutor(max_workers=_acquisition_max_workers(concepts)) as executor:
        for idx, question in enumerate(followup_questions, start=1):
            print(f"{idx}. {question}")
            futures = [
                executor.submit(agent.ask_llm, question, concept)
                for concept in concepts
            ]
            results = [future.result() for future in futures]
            # Each tally counts only replies with the matching prefix, so
            # `probs_yes` and `probs_no` are tracked independently.
            yes_count = sum(
                1 for r in results if r.lower().strip().startswith("yes")
            )
            no_count = sum(1 for r in results if r.lower().strip().startswith("no"))
            probs_yes = yes_count / len(concepts)
            probs_no = no_count / len(concepts)

            ensemble_entropy = _safe_entropy_term(probs_yes) + _safe_entropy_term(
                probs_no
            )
            print("ensemble entropy: ", ensemble_entropy)
            ensemble_entropies.append(ensemble_entropy)
    ensemble_entropies = np.array(ensemble_entropies)
    idx = np.argmax(ensemble_entropies)
    print(f"Maximum Ensemble Entropy at Question {idx + 1}")
    return followup_questions[idx], ensemble_entropies[idx]


def question_ensemble_entropy(
    concepts, followup_questions, agent, item=None, step=None
):
    ensemble_entropies = []
    with ThreadPoolExecutor(max_workers=_acquisition_max_workers(concepts)) as executor:
        for idx, question in enumerate(followup_questions, start=1):
            print(f"{idx}. {question}")
            futures = [
                executor.submit(agent.yes_no_probs, question, concept)
                for concept in concepts
            ]
            results = [future.result() for future in futures]
            probs_yes = [result[0] for result in results]
            ensemble_entropy = ensemble_binary_entropy(probs_yes)
            print("ensemble entropy: ", ensemble_entropy)
            ensemble_entropies.append(ensemble_entropy)
    ensemble_entropies = np.array(ensemble_entropies)
    idx = np.argmax(ensemble_entropies)
    print(f"Maximum Ensemble Entropy at Question {idx + 1}")
    return followup_questions[idx], ensemble_entropies[idx]


def question_eig(
    concepts, followup_questions, agent, item=None, step=None, use_history=False
):
    if len(followup_questions) == 0:
        raise ValueError("question_eig needs at least one question")
    if len(concepts) == 0:
        # With no concepts every EIG term is 0.0, so the first question is
        # returned.
        print(f"Empty concept pool at step {step} for item {item}")
        return followup_questions[0], 0.0
    info_gains = []
    average_entropies = np.zeros(len(followup_questions))
    with ThreadPoolExecutor(max_workers=_acquisition_max_workers(concepts)) as executor:
        for idx, question in enumerate(followup_questions, start=1):
            print(f"{idx}. {question}")
            futures = [
                executor.submit(
                    agent.yes_no_probs, question, concept, use_history=use_history
                )
                for concept in concepts
            ]
            results = [future.result() for future in futures]
            average_entropy = average_binary_entropy(results)
            print("average entropy: ", average_entropy)
            average_entropies[idx - 1] = average_entropy
            probs_yes = [result[0] for result in results]
            ensemble_entropy = ensemble_binary_entropy(probs_yes)
            eig = ensemble_entropy - average_entropy
            print(f"EIG for question {idx}: ", eig)
            info_gains.append(eig)
    info_gains = np.array(info_gains)
    idx = np.argmax(info_gains)
    print(f"Maximum EIG at Question {idx + 1}")
    print(f"Average entropies at step {step}: {average_entropies}")
    # ``wandb.log`` raises before ``wandb.init``, so skip the histogram when
    # no run is active (``--debug`` never starts one).
    if wandb.run is not None:
        w_hist = wandb.Histogram(average_entropies)
        wandb.log({"custom_step": step, f"average_entropy_hist/{item}": w_hist})
    return followup_questions[idx], info_gains[idx]


def _is_yes(text: str) -> int:
    return 1 if str(text).strip().lower().startswith("y") else 0


def _sample_once(agent, question, concept, k):
    """One stochastic sample of the answer to ``question`` under ``concept``."""
    return _is_yes(agent.ask_llm(question, concept, seed=k))


def _np_binary_entropy(p, eps=1e-12):
    """Binary entropy in NATS (natural log), vectorized over ``p``.

    Distinct from `_safe_entropy_term` / `average_binary_entropy` / `ensemble_binary_entropy`
    above, which work in BITS over ``(p_yes, p_no)`` pairs. Only
    `question_eig_sampled` (the ``Sample_EIG`` acquisition) uses this
    nats-based family; the two are not interchangeable, since swapping the log
    base rescales every logged EIG by ``1 / ln 2``.
    """
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _np_average_binary_entropy(per_concept_p):
    per_concept_p = np.asarray(per_concept_p, dtype=float)
    return float(np.mean(_np_binary_entropy(per_concept_p)))


def _np_ensemble_entropy(per_concept_p):
    p_hat = float(np.mean(per_concept_p))
    return float(_np_binary_entropy(p_hat))


def question_eig_sampled(
    concepts,
    followup_questions,
    agent,
    item=None,
    step=None,
    K: int = 5,
    smoothing: float = 0.5,
    max_workers: int = None,
):
    """
    Compute EIG for each followup question using only hard labels from agent.ask_llm.

    Args:
        concepts: iterable of posterior samples θ^(n)
        followup_questions: iterable of questions x_t
        agent: object exposing ask_llm(question, concept, [seed]) -> "yes"/"no"
        item: unused, but required — ``FunctionQuestionChooser.choose`` sends it
        step: unused, but required — ``FunctionQuestionChooser.choose`` sends it
        K: # stochastic repeats per (question, concept) to estimate p_yes
        smoothing: Laplace smoothing for stability
        max_workers: ThreadPoolExecutor cap (default: min(32, len(concepts)*K))

    Returns:
        (best_question, best_eig)
    """
    N = len(concepts)
    if N == 0 or len(followup_questions) == 0:
        raise ValueError("Need at least one concept and one question.")

    if max_workers is None:
        max_workers = max(1, min(32, N * K))

    info_gains = np.zeros(len(followup_questions), dtype=float)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for qi, q in enumerate(followup_questions):
            futures = []
            for c_idx, concept in enumerate(concepts):
                for k in range(K):
                    futures.append(
                        (c_idx, executor.submit(_sample_once, agent, q, concept, k))
                    )

            yes_counts = np.zeros(N, dtype=float)
            for c_idx, f in futures:
                yes_counts[c_idx] += f.result()

            per_concept_p = (yes_counts + smoothing) / (K + 2.0 * smoothing)

            avg_ent = _np_average_binary_entropy(per_concept_p)
            ens_ent = _np_ensemble_entropy(per_concept_p)
            info_gains[qi] = ens_ent - avg_ent
            print(f"EIG for question {qi + 1}: ", info_gains[qi])

    best_idx = int(np.argmax(info_gains))
    return followup_questions[best_idx], float(info_gains[best_idx])


def question_split_score(
    concepts,
    followup_questions,
    agent,
    item=None,
    step=None,
    original=True,
    base=False,
    logprobs=False,
    renormalize=False,
):
    N = len(concepts)
    split_scores = []

    # When logprobs flag is set, pre-compute log-probabilities for each concept
    # aligned with the ordering of `concepts` using the agent's guesser model.
    concept_logprobs = None
    concept_probs = None
    if logprobs and N > 0:
        concept_probs = _concept_probs_given_history(
            agent, getattr(agent, "guesser_messages", []), concepts
        )
        concept_logprobs = np.log(concept_probs + 1e-10)
        print("Concept probabilities from guesser model:", concept_probs)

    # Loop-invariant, so computed once. Both branches only rebind `probas`; the
    # per-question code never writes into it.
    if logprobs and concept_logprobs is not None:
        # Use precomputed concept log-probabilities aligned with `concepts`
        probas = np.exp(concept_logprobs)
    else:
        probas = np.ones(N)
    probas = probas / (probas.sum() + 1e-9)

    with ThreadPoolExecutor(max_workers=_acquisition_max_workers(concepts)) as executor:
        for idx, question in enumerate(followup_questions, start=1):
            print(f"{idx}. {question}")
            futures = [
                executor.submit(agent.ask_llm, question, concept)
                for concept in concepts
            ]
            results = [future.result() for future in futures]

            # Define this_statement based on results
            this_statement = [
                result.lower().strip().startswith("yes") for result in results
            ]

            if base:
                score = abs(N / 2 - np.sum(this_statement))
            elif renormalize:
                H_true = np.array([p for p, a in zip(probas, this_statement) if a])
                H_false = np.array([p for p, a in zip(probas, this_statement) if not a])

                p_yes = float(H_true.sum())
                p_no = float(H_false.sum())

                print(f"p(yes)={p_yes}, p(no)={p_no}, sum={p_yes + p_no}, N={N}")

                H_true_norm = H_true / (p_yes + 1e-10)
                H_false_norm = H_false / (p_no + 1e-10)

                H_theta_given_yes = -np.sum(H_true_norm * np.log(H_true_norm + 1e-8))
                H_theta_given_no = -np.sum(H_false_norm * np.log(H_false_norm + 1e-8))
                E_post = p_yes * H_theta_given_yes + p_no * H_theta_given_no
                score = -E_post
            else:
                print("Not renormalizing")
                H_true = np.array([p for p, a in zip(probas, this_statement) if a])
                H_false = np.array([p for p, a in zip(probas, this_statement) if not a])
                print(
                    f"Number of 'yes' answers: {len(H_true)}, Number of 'no' answers: {len(H_false)}"
                )
                entropy_true = -np.sum(H_true * np.log(H_true + 1e-8))
                entropy_false = -np.sum(H_false * np.log(H_false + 1e-8))
                score = -(entropy_true * len(H_true) + entropy_false * len(H_false))
            print(f"Split score for question {idx}: ", score)
            split_scores.append(score)
    # The two baselines take opposite extremes of split_scores: `original`
    # selects the minimum (ascending argsort), the variant selects the maximum
    # (descending). The argsort form fixes the tie-break order in both cases.
    scores = np.array(split_scores)
    idx_argsort = scores.argsort() if original else scores.argsort()[::-1]
    print("idx_argsort:", idx_argsort)
    best = int(idx_argsort[0])
    print(f"Best Split Score at Question {best + 1}")
    return followup_questions[best], split_scores[best]


def question_theta_eig_union_mixture(
    concepts, followup_questions, agent, item=None, step=None, concept_sampler=None
):
    """Theta-EIG scored over the UNION of the two answer branches (``Data_estimation_union_mixture``).

    For each candidate question:
      - estimate p(Yes), p(No) given history
      - sample concepts conditional on history + question + 'Yes.' and on
        history + question + 'No.'
      - mix the two branch posteriors over the union of their concepts, keyed by
        concept identity, and score
        ``H[mixture] - (p_yes * H[yes branch] + p_no * H[no branch])``

    See :func:`question_theta_eig_rank_aligned` for the variant that combines
    the two branch posteriors by probability rank.

    Returns:
      (best_question, best_eig, yes_concepts_for_best, no_concepts_for_best)
    """
    print("In question_theta_eig_union_mixture")
    info_gains = []
    per_q_yes_concepts = []
    per_q_no_concepts = []

    with ThreadPoolExecutor(max_workers=min(len(followup_questions), 15)) as executor:

        def calculate_eig(idx, question):
            print(f"{idx}. {question}")

            p_yes, p_no = agent.answer_probs_given_history(question, chat=True)

            # Prepare message histories for 'Yes' branch
            agent_messages = agent.guesser_messages.copy()
            agent_messages_cropped = agent_messages[1:]
            print("Messages in agent:", agent_messages)
            agent_messages.append({"role": "assistant", "content": question})
            agent_messages.append({"role": "user", "content": "Yes."})
            agent_messages_cropped.append({"role": "assistant", "content": question})
            agent_messages_cropped.append({"role": "user", "content": "Yes."})
            agent_messages_cropped_yes = agent_messages_cropped.copy()

            # Helper: add branch contribution to mixture and accumulate branch probs
            def _accumulate_branch(
                probs, candidate_concepts, branch_weight, branch_accumulator
            ):
                for i, concept in enumerate(candidate_concepts):
                    theta_probs[concept] += probs[i] * branch_weight
                    branch_accumulator[concept] += probs[i]

            # Sample concepts for 'Yes' response
            yes_concepts = concept_sampler.generate_diverse_samples_from_chat(
                agent_messages,
                stage=step,
                valid_concepts=concepts.copy(),
                task=agent.task,
            )

            # Prepare message histories for 'No' branch (reuse arrays, flip last turn)
            agent_messages_cropped[-1] = {"role": "user", "content": "No."}
            agent_messages[-1] = {"role": "user", "content": "No."}

            no_concepts = concept_sampler.generate_diverse_samples_from_chat(
                agent_messages,
                stage=step,
                valid_concepts=concepts.copy(),
                task=agent.task,
            )

            theta_union = list(set(yes_concepts).union(set(no_concepts)))

            theta_probs = {concept: 0.0 for concept in theta_union}
            yes_theta_probs = {concept: 0.0 for concept in yes_concepts}
            no_theta_probs = {concept: 0.0 for concept in no_concepts}

            # Yes branch probabilities and entropy. `yes_probs` and
            # `average_entropy_yes` are consumed only inside this branch; the EIG
            # below is built from `yes_theta_probs` / `no_theta_probs`.
            if len(yes_concepts) > 0:
                yes_probs = _concept_probs_given_history(
                    agent, agent_messages_cropped_yes, yes_concepts
                )
                print(f"Probabilities for concepts (yes): {yes_probs}")
                average_entropy_yes = -np.sum(yes_probs * np.log2(yes_probs + 1e-10))
                print(f"Average entropy for concepts (yes): {average_entropy_yes}")
                _accumulate_branch(yes_probs, yes_concepts, p_yes, yes_theta_probs)

            # No branch probabilities and entropy
            if len(no_concepts) > 0:
                no_probs = _concept_probs_given_history(
                    agent, agent_messages_cropped, no_concepts
                )
                print(f"Probabilities for concepts (no): {no_probs}")
                average_entropy_no = -np.sum(no_probs * np.log2(no_probs + 1e-10))
                print(f"Average entropy for concepts (no): {average_entropy_no}")
                _accumulate_branch(no_probs, no_concepts, p_no, no_theta_probs)

            print("Theta_probs:", theta_probs)
            print("Sum of theta_probs:", sum(theta_probs.values()))

            # calculate entropy of theta_probs
            overall_probs = np.array(list(theta_probs.values()))
            overall_probs /= np.sum(overall_probs) + 1e-10
            overall_entropy = -np.sum(overall_probs * np.log2(overall_probs + 1e-10))
            print(f"Overall entropy: {overall_entropy}")

            # Entropies from accumulated branch distributions
            yes_theta_probs_array = np.array(list(yes_theta_probs.values()))
            if len(yes_theta_probs_array) > 0:
                yes_entropy = -np.sum(
                    yes_theta_probs_array * np.log2(yes_theta_probs_array + 1e-10)
                )
            else:
                yes_entropy = 0.0

            no_theta_probs_array = np.array(list(no_theta_probs.values()))
            if len(no_theta_probs_array) > 0:
                no_entropy = -np.sum(
                    no_theta_probs_array * np.log2(no_theta_probs_array + 1e-10)
                )
            else:
                no_entropy = 0.0

            average_entropy = p_yes * yes_entropy + p_no * no_entropy
            print(f"Weighted average entropy from theta_probs: {average_entropy}")

            print("new calculation of eig: ", overall_entropy - average_entropy)

            return overall_entropy - average_entropy, yes_concepts, no_concepts

        futures = [
            executor.submit(calculate_eig, idx, question)
            for idx, question in enumerate(followup_questions, start=1)
        ]

        for future in futures:
            eig_val, yes_c, no_c = future.result()
            info_gains.append(eig_val)
            per_q_yes_concepts.append(yes_c)
            per_q_no_concepts.append(no_c)

    info_gains = np.array(info_gains)
    idx = np.argmax(info_gains)
    print(f"Maximum EIG at Question {idx + 1}, EIG: {info_gains[idx]}")

    return (
        followup_questions[idx],
        info_gains[idx],
        per_q_yes_concepts[idx],
        per_q_no_concepts[idx],
    )


def question_theta_eig_rank_aligned(
    concepts, followup_questions, agent, item=None, step=None, concept_sampler=None
):
    """Theta-EIG with the two answer branches paired by RANK (``Data_estimation_rank_aligned``).

    For each candidate question:
      - estimate p(Yes), p(No) given history
      - sample concepts conditional on history + question + 'Yes.' and on
        history + question + 'No.'
      - sort both branch posteriors by descending probability, truncate both to
        the shorter one, renormalize, and mix them position by position

    The mixture is formed over the leading ``min_len`` entries of each branch,
    so the Yes and No posteriors are combined by probability rank. See
    :func:`question_theta_eig_union_mixture` for the variant that combines them
    over the union of concepts.

    Returns:
      (best_question, best_eig, yes_concepts_for_best, no_concepts_for_best)
    """
    print("In question_theta_eig_rank_aligned")
    info_gains = []
    per_q_yes_concepts = []
    per_q_no_concepts = []

    with ThreadPoolExecutor(max_workers=min(len(followup_questions), 15)) as executor:

        def calculate_eig(idx, question):
            print(f"{idx}. {question}")

            # Calculate p(yes) and p(no) given history
            p_yes, p_no = agent.answer_probs_given_history(question, chat=True)

            # Prepare message histories for 'Yes' branch
            agent_messages = agent.guesser_messages.copy()
            agent_messages_cropped = agent_messages[1:]
            print("Messages in agent:", agent_messages)
            agent_messages.append({"role": "assistant", "content": question})
            agent_messages.append({"role": "user", "content": "Yes."})
            agent_messages_cropped.append({"role": "assistant", "content": question})
            agent_messages_cropped.append({"role": "user", "content": "Yes."})
            agent_messages_cropped_yes = agent_messages_cropped.copy()

            # Sample concepts for 'Yes' response
            yes_concepts = concept_sampler.generate_diverse_samples_from_chat(
                agent_messages,
                stage=step,
                valid_concepts=concepts.copy(),
                task=agent.task,
            )

            # `min_len` and `weighted_entropy` below read `yes_probs` and
            # `average_entropy_yes`, so both are set on either path.
            if len(yes_concepts) == 0:
                yes_probs = np.array([])
                average_entropy_yes = 0.0
            else:
                yes_probs = _concept_probs_given_history(
                    agent, agent_messages_cropped_yes, yes_concepts
                )
                print(f"Probabilities for concepts (yes): {yes_probs}")
                average_entropy_yes = -np.sum(yes_probs * np.log2(yes_probs + 1e-10))
                print(f"Average entropy for concepts (yes): {average_entropy_yes}")

            # Prepare message histories for 'No' branch (reuse arrays, flip last turn)
            agent_messages_cropped[-1] = {"role": "user", "content": "No."}
            agent_messages[-1] = {"role": "user", "content": "No."}

            no_concepts = concept_sampler.generate_diverse_samples_from_chat(
                agent_messages,
                stage=step,
                valid_concepts=concepts.copy(),
                task=agent.task,
            )

            if len(no_concepts) == 0:
                no_probs = np.array([])
                average_entropy_no = 0.0
            else:
                no_probs = _concept_probs_given_history(
                    agent, agent_messages_cropped, no_concepts
                )
                print(f"Probabilities for concepts (no): {no_probs}")
                average_entropy_no = -np.sum(no_probs * np.log2(no_probs + 1e-10))
                print(f"Average entropy for concepts (no): {average_entropy_no}")

            min_len = min(len(yes_probs), len(no_probs))
            if min_len > 0:
                yes_indices = np.argsort(-yes_probs)
                no_indices = np.argsort(-no_probs)
                yes_probs_sorted = yes_probs[yes_indices[:min_len]]
                no_probs_sorted = no_probs[no_indices[:min_len]]
                yes_probs_norm = yes_probs_sorted / (np.sum(yes_probs_sorted) + 1e-10)
                no_probs_norm = no_probs_sorted / (np.sum(no_probs_sorted) + 1e-10)
                weighted_avg_probs = p_yes * yes_probs_norm + p_no * no_probs_norm
                weighted_avg_entropy = -np.sum(
                    weighted_avg_probs * np.log2(weighted_avg_probs + 1e-10)
                )
            else:
                weighted_avg_entropy = 0.0
            print(f"Entropy of weighted average distribution: {weighted_avg_entropy}")

            weighted_entropy = p_yes * average_entropy_yes + p_no * average_entropy_no

            weighted_eig = weighted_avg_entropy - weighted_entropy
            print(f"EIG for question {idx}: ", weighted_eig)

            return weighted_eig, yes_concepts, no_concepts

        futures = [
            executor.submit(calculate_eig, idx, question)
            for idx, question in enumerate(followup_questions, start=1)
        ]

        for future in futures:
            eig_val, yes_c, no_c = future.result()
            info_gains.append(eig_val)
            per_q_yes_concepts.append(yes_c)
            per_q_no_concepts.append(no_c)

    info_gains = np.array(info_gains)
    idx = np.argmax(info_gains)
    print(f"Maximum EIG at Question {idx + 1}")

    return (
        followup_questions[idx],
        info_gains[idx],
        per_q_yes_concepts[idx],
        per_q_no_concepts[idx],
    )


def average_binary_entropy(probs):
    """Return the mean binary entropy in bits over a list of (p_yes, p_no) pairs.

    Each term goes through ``_safe_entropy_term``, the same guard
    ``ensemble_binary_entropy`` uses: both halves of one EIG must treat the
    p in {0, 1} limits the same way, and a bare ``math.log2(p)`` raises a math
    domain error when a backend reports an exactly-zero probability.
    """
    entropies = [
        _safe_entropy_term(p_yes) + _safe_entropy_term(p_no) for p_yes, p_no in probs
    ]
    if len(entropies) == 0:
        return 0.0
    return sum(entropies) / len(entropies)


def get_animal_concept_prompt(num_samples, init=False):
    prompt_init = (
        f"Generate a list of {num_samples} diverse, unique, representative examples of animals. Some of these examples should be more obscure. "
        "Each example should be on a separate line and be a single distinct entity with nothing else added (e.g., 'Dog', 'Siberian Flying Squirrel', 'Javan Rhiniceros', 'Chinchilla'), and no repetition."
    )
    prompt = (
        "You are participating in a game of 20 questions."
        f"Based on the information provided in the questions and answers above, generate a list of **up to** {num_samples} diverse, unique examples of animals. Each animal must be consistent with *all* the established clues from the conversation. "
        "Each example should be on a separate line and be a single distinct entity with no extra text added (e.g., 'Dog', 'Siberian Flying Squirrel', 'Javan Rhiniceros', 'Chinchilla'), and no repetition. "
        "Do not include any additional commentary or explanations - **only** provide an animal on each line. "
        "Ensure diversity in terms of species, genus, family, class etc., while strictly adhering to the constraints set by the previous questions and answers. Some of the examples should be more obscure, including specific species within families."
    )
    if init:
        return prompt_init
    return prompt


def get_celebrity_question_prompt(num_questions, concepts, final=False):
    """Build the question-generation prompt for the ``celebrities`` dataset.

    All three datasets resolve their question prompt through the one
    ``(num_questions, concepts, final)`` signature that
    :func:`get_animal_question_prompt` and :func:`get_thing_question_prompt`
    also use, so this person-specific wording cannot reach an animals or a
    things game.
    """
    prompt = (
        f"You are participating in a game of 20 questions to deduce the identity of a famous person. Based on the conversation so far, generate a list of **up to** {num_questions} diverse, strategic and insightful follow-up questions that will help efficiently narrow down the identity of the person. "
        f"Given the questions and answers asked, another LLM has generated a list of possible identities: {concepts}. It may be good to attempt to narrow these down. "
        "The questions must be strictly binary - each question can only be answered with 'Yes' or 'No'. Additionally, some of the questions should attempt to guess the identity "
        "of the item (for example, 'Is it item X?') while still being answerable in a binary fashion. "
        "Return each question on a separate line with no extra commentary or numbering. "
        "Each question must be consistent with all previous answers and should aim to maximize information gain by distinguishing between the remaining possibilities. Avoid redundancy and ensure that the questions cover diverse attributes such as time period, profession, nationality, and notable achievements. "
    )

    prompt_final = (
        "You are participating in a game of 20 questions to deduce the identity of a famous person. They can be real or fictional and may be from any time period, including the present day. "
        f"Given the questions and answers, another LLM has generated a list of possible identities which may be incomplete: {concepts}. "
        "Based on the conversation, guess the identity of the person. Just respond with the name of the person and nothing else. "
    )
    if final:
        return prompt_final
    return prompt


def get_animal_question_prompt(num_questions, concepts, final=False):
    prompt = (
        f"You are participating in a game of 20 questions to deduce the identity of a famous animal. Based on the conversation so far, generate a list of **up to** {num_questions} diverse, strategic and insightful follow-up questions that will help efficiently narrow down the animal. "
        f"Given the questions and answers asked, another LLM has generated a list of possible identities: {concepts}. It may be good to attempt to narrow these down. "
        "The questions must be strictly binary - each question can only be answered with 'Yes' or 'No'. Additionally, some of the questions should attempt to guess the identity "
        "of the item (for example, 'Is it item X?') while still being answerable in a binary fashion. "
        "Return each question on a separate line with no extra commentary or numbering. "
        "Each question must be consistent with all previous answers and should aim to maximize information gain by distinguishing between the remaining possibilities. Avoid redundancy and ensure that the questions cover diverse attributes such as time period, profession, nationality, and notable achievements. "
    )

    prompt_final = (
        "You are participating in a game of 20 questions to deduce an animal. "
        f"Given the questions and answers, another LLM has generated a list of possible animals which may be incomplete: {concepts}. "
        "Based on the conversation, guess the animal. Just respond with the name of the animal and nothing else. "
    )
    if final:
        return prompt_final
    return prompt


def get_thing_question_prompt(num_questions, concepts, final=False):
    prompt = (
        f"You are participating in a game of 20 questions to deduce an entity. Based on the conversation so far, generate a list of **up to** {num_questions} diverse, strategic and insightful follow-up questions that will help efficiently narrow down what the entity is. "
        f"Given the questions and answers asked, another LLM has generated a list of possible entities: {concepts}. It may be good to attempt to narrow these down. "
        "The questions must be strictly binary - each question can only be answered with 'Yes' or 'No'. Additionally, some of the questions should attempt to guess the identity "
        "of the item (for example, 'Is it item X?') while still being answerable in a binary fashion. "
        "Return each question on a separate line with no extra commentary or numbering. "
        "Each question must be consistent with all previous answers and should aim to maximize information gain by distinguishing between the remaining possibilities. Avoid redundancy and ensure that the questions cover diverse attributes such as size, shape and existence. "
    )

    prompt_final = (
        "You are participating in a game of 20 questions to deduce an entity. "
        f"Given the questions and answers, another LLM has generated a list of possible entities which may be incomplete: {concepts}. "
        "Based on the conversation, guess the entity. Just respond with the name of the entity and nothing else. "
    )
    if final:
        return prompt_final
    return prompt


def text_mentions_item(item, question):
    # Escape each alternative so items containing regex metacharacters
    # (e.g. "J.K. Rowling") are matched literally rather than as patterns.
    return any(
        re.search(rf"(?:^|\W){re.escape(i.strip().lower())}(?:$|\W)", question.lower())
        for i in item.lower().split("|")
    )


def parse_llm_line_list(text, *, boundary_prefixes, clean_line):
    """Parse a newline-delimited list out of raw LLM output.

    Keeps the text after the last ``</think>``, drops everything up to and
    including the first line whose stripped form starts with one of
    ``boundary_prefixes``, then applies ``clean_line`` to each remaining line
    and keeps the truthy results. ``clean_line`` may return ``""`` to drop a
    line.

    ``QuestionSampler.generate_followup_questions`` and
    ``ConceptSampler._parse_concepts`` share this skeleton and supply their own
    (intentionally different) per-line cleaning.
    """
    body = text.split("</think>")[-1]
    lines = body.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if any(line.strip().startswith(p) for p in boundary_prefixes):
            start = i + 1
            break
    out = []
    for line in lines[start:]:
        cleaned = clean_line(line)
        if cleaned:
            out.append(cleaned)
    return out


def extract_action_content(llm_output: str):
    """Return ``(action_type, content)`` from ``ASK["..."]`` or ``GUESS["..."]``."""
    match = re.search(r'(ASK|GUESS)\["(.+?)"\]', llm_output)
    if match:
        action_type = match.group(1)
        content = match.group(2).strip()
        return action_type, content
    return None, None
