"""Unit tests for sequence generation (numpy only, no model needed).

    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perturb.sequences import build_candidate_pool, sample_batch  # noqa: E402

VOCAB = 1000


def fake_log_probs():
    # token i has score i, so token 0 is least likely and token 999 most likely
    return np.arange(VOCAB, dtype=float)


def test_pool_drops_both_ends():
    pool = build_candidate_pool(fake_log_probs(), drop_top_frac=0.1, drop_bottom_frac=0.2)
    assert pool.min() == 200 and pool.max() == 899
    assert len(pool) == 700


def test_pool_excludes_ids():
    pool = build_candidate_pool(fake_log_probs(), 0.0, 0.0, exclude_ids=[5, 6])
    assert 5 not in pool and 6 not in pool and len(pool) == VOCAB - 2


def test_pool_rejects_bad_fractions():
    with pytest.raises(ValueError):
        build_candidate_pool(fake_log_probs(), 0.6, 0.5)


def test_batch_layout_and_no_repeats():
    pool = build_candidate_pool(fake_log_probs(), 0.01, 0.25)
    clause = [500, 501, 502]
    b = sample_batch(np.random.default_rng(0), pool, clause, 32, 32, batch_size=64)
    L = 32 + 3 + 32
    assert b.baseline.shape == b.perturbed.shape == (64, L)
    assert (b.pert_start, b.pert_end) == (32, 35)
    for base, pert in zip(b.baseline, b.perturbed):
        assert len(set(base)) == L          # no repeats in baseline
        assert len(set(pert)) == L          # no repeats in perturbed
        assert (base[:32] == pert[:32]).all()   # shared prefix
        assert (base[35:] == pert[35:]).all()   # shared suffix
        assert list(pert[32:35]) == clause      # clause in the middle
        assert not set(base[32:35]) & set(clause)  # filler never equals clause tokens
        assert set(base).issubset(set(pool))


def test_reproducible_with_seed():
    pool = build_candidate_pool(fake_log_probs(), 0.01, 0.25)
    a = sample_batch(np.random.default_rng(7), pool, [500], 8, 8, 4)
    b = sample_batch(np.random.default_rng(7), pool, [500], 8, 8, 4)
    assert (a.baseline == b.baseline).all() and (a.perturbed == b.perturbed).all()


def test_pool_too_small():
    pool = np.arange(10)
    with pytest.raises(ValueError):
        sample_batch(np.random.default_rng(0), pool, [3], 8, 8, 1)
