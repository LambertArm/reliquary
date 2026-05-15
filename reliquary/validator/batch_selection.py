"""Drand-anchored batch selection + emission distribution.

Called once per window at seal time. Two things happen here:

1. Pick the B distinct prompts that go into the GRPO training step. Order
   is driven by a drand round whose σ is published *after* window close,
   so miners cannot grind submissions against it. TCP arrival time is no
   longer used — being co-located with the validator gives no edge.

2. Compute a uniform reward distribution across all GRAIL-validated
   submissions whose prompt landed in the winning set. Multiple miners
   per prompt is now allowed (see ``MAX_SUBMISSIONS_PER_PROMPT``); they
   split that prompt's slot share. A miner sybiling the same prompt with
   N hotkeys earns N × (slot_share / N) = slot_share — identical to a
   single hotkey, so spawning extra hotkeys is strictly wasteful.

v2.3+: replaces the v2.2 pure-FIFO ``select_batch``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Protocol

from reliquary.validator.cooldown import CooldownMap


class _SubmissionLike(Protocol):
    """Duck-typed submission — works with any class exposing these attrs."""

    hotkey: str
    prompt_idx: int
    merkle_root: bytes


def _prompt_order_key(ordering_seed: bytes, prompt_idx: int) -> bytes:
    """Hash key used to drand-order prompts at seal time."""
    h = hashlib.sha256()
    h.update(ordering_seed)
    h.update(prompt_idx.to_bytes(8, "big", signed=False))
    return h.digest()


def _training_pick_seed(ordering_seed: bytes, prompt_idx: int) -> int:
    """Per-prompt seed for selecting which submission feeds training.

    Bound to ``ordering_seed`` (unknown to miners during the window) and
    to ``prompt_idx`` (so different prompts get different picks). Returned
    as an int so ``random.Random`` is fully deterministic across CPython
    versions and platforms.
    """
    h = hashlib.sha256()
    h.update(ordering_seed)
    h.update(prompt_idx.to_bytes(8, "big", signed=False))
    h.update(b"train")
    return int.from_bytes(h.digest(), "big")


def _canonical_submission_key(sub: _SubmissionLike) -> tuple[str, bytes]:
    """Sort key that any validator computes identically.

    Submissions arrive in network-dependent order, so the in-memory list
    per prompt differs across validators. Sorting before ``random.choice``
    makes the training pick consensus-safe.
    """
    return (sub.hotkey, sub.merkle_root)


def select_batch_and_distribute(
    submissions_by_prompt: dict[int, list[Any]],
    *,
    b: int,
    ordering_seed: bytes,
    cooldown_map: CooldownMap,
    current_window: int,
    pool: float = 1.0,
) -> tuple[list[Any], dict[str, float]]:
    """Pick the training batch and the reward distribution.

    Args:
        submissions_by_prompt: map of ``prompt_idx`` → list of
            GRAIL-validated submissions for that prompt (each list size in
            [1, MAX_SUBMISSIONS_PER_PROMPT]).
        b: training batch size (= B_BATCH).
        ordering_seed: drand-derived seed for the post-close round.
            Unknown to miners during the submission window.
        cooldown_map: read-only view of which prompts are in cooldown.
            Prompts in cooldown are excluded from the winning set entirely
            (consistent with v2.2 semantics).
        current_window: needed to evaluate cooldown.
        pool: total emission budget for the window. Defaults to 1.0 so
            rewards sum to 1 (caller can scale).

    Returns:
        (training_batch, rewards_by_hotkey)
        - training_batch: up to ``b`` ValidSubmission objects, one per
          winning prompt, picked by drand-seeded random among the prompt's
          candidates.
        - rewards_by_hotkey: dict hotkey → float, summing to ``pool``
          (modulo float roundoff) when at least one winning prompt exists.

    Does NOT mutate ``cooldown_map`` — the caller records post-selection.
    """
    if b <= 0 or not submissions_by_prompt:
        return [], {}

    eligible_prompts = [
        p for p in submissions_by_prompt
        if not cooldown_map.is_in_cooldown(p, current_window)
    ]
    if not eligible_prompts:
        return [], {}

    eligible_prompts.sort(key=lambda p: _prompt_order_key(ordering_seed, p))
    winning_prompts = eligible_prompts[:b]
    n_winners = len(winning_prompts)
    slot_share = pool / n_winners

    training_batch: list[Any] = []
    rewards: dict[str, float] = {}
    for prompt_idx in winning_prompts:
        candidates = sorted(
            submissions_by_prompt[prompt_idx], key=_canonical_submission_key,
        )
        k_p = len(candidates)
        per_miner = slot_share / k_p
        for sub in candidates:
            rewards[sub.hotkey] = rewards.get(sub.hotkey, 0.0) + per_miner
        rng = random.Random(_training_pick_seed(ordering_seed, prompt_idx))
        training_batch.append(rng.choice(candidates))

    return training_batch, rewards
