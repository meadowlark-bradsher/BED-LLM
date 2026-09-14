"""Per-dataset prompt text shared across the acquisition and logprob paths.

This is a leaf module (no intra-package imports) so that both ``utils`` and
``theta_eig`` can read the same strings. Keeping one copy matters: the
completion prompt is part of the conditioning distribution, so the acquisition
functions in ``utils`` and the chat-logprob path in ``theta_eig`` must ask the
guesser for an entity in exactly the same words, byte for byte.
"""

PROMPT = """Return only the full name of one randomly selected famous person (living or deceased) consistent with the questions and answers above.

To increase randomness:
1. Internally brainstorm a pool of diverse and representative individuals.
2. Avoid defaulting to the most globally ubiquitous celebrities or famous figures.

Output rules:
- Output ONLY the person's full name, nothing else.
- No extra words, explanations, numbering, or punctuation beyond what's in the name itself (hyphens/apostrophes allowed if part of the name)."""

_ANIMALS_PROMPT = (
    "Return only the exact name of one animal consistent with the questions and answers above.\n\n"
    "To increase randomness:\n"
    "1. Internally brainstorm a pool of diverse and representative animals.\n"
    "2. Avoid defaulting to the most obvious animals.\n"
    "Output rules:\n"
    "- Output ONLY the animal name, nothing else.\n"
    "- No extra words, explanations, numbering, or punctuation beyond what's in the name itself."
)

_THINGS_PROMPT = (
    "Return only the exact name of one entity consistent with the questions and answers above.\n\n"
    "To increase randomness:\n"
    "1. Internally brainstorm a pool of diverse and representative entities/objects.\n"
    "2. Avoid defaulting to the most obvious or common entities.\n\n"
    "Output rules:\n"
    "- Output ONLY the entity name, nothing else.\n"
    "- No extra words, explanations, numbering, or punctuation beyond what's in the name itself."
)

# Each dataset has its own wording. The exact text is part of the conditioning
# distribution, so these strings are treated as fixed.
_COMPLETION_PROMPTS = {
    "animals": _ANIMALS_PROMPT,
    "things": _THINGS_PROMPT,
    "celebrities": PROMPT,
}


def task_completion_prompt(task) -> str:
    """Return the completion prompt that asks for one entity of ``task``.

    Shared by ``question_split_score``,
    ``question_theta_eig_union_mixture``, ``question_theta_eig_rank_aligned`` and
    ``theta_eig.item_probs_from_chat_logprobs``.
    An unknown task falls back to the celebrities prompt.
    """
    task_l = task.lower() if isinstance(task, str) else "celebrities"
    return _COMPLETION_PROMPTS.get(task_l, PROMPT)
