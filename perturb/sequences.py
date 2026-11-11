"""Random-token sequence generation.

Pure numpy, no torch, so it can be unit-tested without loading a model
(see tests/test_sequences.py).

Layout of one sequence pair (BOS is added later, in experiment.py):

    baseline : [prefix (n_prefix)] [filler  (k)] [suffix (n_suffix)]
    perturbed: [prefix (n_prefix)] [clause  (k)] [suffix (n_suffix)]

* prefix and suffix are IDENTICAL in both sequences.
* `filler` is k random tokens, so the baseline is fully random and has the
  same length as the perturbed one. That keeps every suffix token at the same
  absolute position in both runs, so the residual-stream delta at a suffix
  position is caused only by what sits in the middle slot.
* No token appears twice within a sequence: prefix, filler and suffix are drawn
  without replacement, and the clause's own tokens are removed from the pool.
  (A repeated token would let induction heads fire and confound the result.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_candidate_pool(
    log_probs: np.ndarray,
    drop_top_frac: float,
    drop_bottom_frac: float,
    exclude_ids: list[int] | tuple[int, ...] = (),
) -> np.ndarray:
    """Token ids left after dropping the most- and least-likely tokens.

    log_probs: (vocab,) score per token; higher = more likely.
    Returns a sorted array of token ids.
    """
    # Guard against nonsense settings: each fraction must be in [0, 1) and
    # together they must leave at least some tokens.
    if not (0 <= drop_top_frac < 1 and 0 <= drop_bottom_frac < 1):
        raise ValueError("drop fractions must be in [0, 1)")
    if drop_top_frac + drop_bottom_frac >= 1:
        raise ValueError("dropping that much leaves no tokens")

    vocab = len(log_probs)
    # argsort ascending: order[0] is the LEAST likely token, order[-1] the MOST.
    # kind="stable" keeps ties in id order so results are deterministic.
    order = np.argsort(log_probs, kind="stable")
    # Convert fractions to token counts (e.g. 0.25 * 50257 -> 12564 tokens).
    n_bottom = int(round(drop_bottom_frac * vocab))
    n_top = int(round(drop_top_frac * vocab))
    # Slice off both ends of the ranking; what's left is the middle band.
    keep = order[n_bottom : vocab - n_top]
    # Remove specific ids we never want (GPT-2's BOS/EOS token, id 50256).
    if len(exclude_ids):
        keep = keep[~np.isin(keep, np.asarray(exclude_ids))]
    # Sorted so the pool is the same regardless of tie order, which makes the
    # rng draws reproducible run to run.
    return np.sort(keep)


@dataclass
class SequenceBatch:
    """A batch of matched (baseline, perturbed) pairs, BOS not yet added."""

    baseline: np.ndarray   # (B, L) token ids, no BOS
    perturbed: np.ndarray  # (B, L) token ids, no BOS
    pert_start: int        # index where the middle slot starts (no BOS)
    pert_end: int          # exclusive end of the middle slot


def sample_batch(
    rng: np.random.Generator,
    pool: np.ndarray,
    clause_ids: list[int] | np.ndarray,
    n_prefix: int,
    n_suffix: int,
    batch_size: int,
) -> SequenceBatch:
    """Sample `batch_size` (baseline, perturbed) pairs. See module docstring."""
    clause_ids = np.asarray(clause_ids, dtype=np.int64)
    k = len(clause_ids)  # length of the middle slot, e.g. 3 for " to the store"
    if k == 0:
        raise ValueError("perturbation clause tokenized to zero tokens")

    # Remove clause tokens from the pool so the random parts can never repeat
    # a clause token (keeps the "no repeated tokens" rule for the perturbed seq).
    usable = pool[~np.isin(pool, clause_ids)]
    # One sequence needs this many DISTINCT random tokens:
    # prefix + filler (same length as the clause) + suffix.
    needed = n_prefix + k + n_suffix
    if needed > len(usable):
        raise ValueError(f"need {needed} distinct tokens, pool only has {len(usable)}")

    L = needed  # total sequence length (without BOS)
    baseline = np.empty((batch_size, L), dtype=np.int64)
    perturbed = np.empty((batch_size, L), dtype=np.int64)
    s, e = n_prefix, n_prefix + k  # middle slot = [s, e)
    for b in range(batch_size):
        # Single draw WITHOUT replacement -> every token in it is unique.
        # Uniform over the pool, as the professor specified.
        draw = rng.choice(usable, size=needed, replace=False)
        prefix, filler, suffix = draw[:s], draw[s:e], draw[e:]
        # Baseline: random everywhere (filler in the middle slot).
        baseline[b] = np.concatenate([prefix, filler, suffix])
        # Perturbed: SAME prefix and suffix, clause in the middle slot.
        perturbed[b] = np.concatenate([prefix, clause_ids, suffix])

    return SequenceBatch(baseline, perturbed, pert_start=s, pert_end=e)
