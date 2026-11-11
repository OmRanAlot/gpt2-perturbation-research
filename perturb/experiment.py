"""Core experiment: residual-stream delta between baseline and perturbed runs.

For each perturbation clause:
  1. Sample N (baseline, perturbed) pairs (see sequences.py for the layout).
  2. Run both through the model, caching the residual stream at every depth.
  3. At each suffix position, compute how far the perturbed residual moved
     away from the baseline residual:
        abs     = ||r_pert - r_base||_2
        rel     = abs / ||r_base||_2      (residual norm grows with depth, so this
                                            makes layers comparable)
        cosdist = 1 - cos(r_pert, r_base)
  4. Index suffix positions by distance from the END of the perturbation:
     the first suffix token is distance 1.

Saved per clause: arrays of shape (n_depths, n_trials, n_suffix).
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .model import bos_log_probs, load_model, resid_hook_names, resid_labels
from .sequences import build_candidate_pool, sample_batch


def slugify(text: str) -> str:
    # " to the store" -> "to_the_store": safe to use as a file name.
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "clause"


@torch.no_grad()
def cache_residuals(model, tokens: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Run the model once and stack the residual stream: (n_depths, B, L, d_model)."""
    # Only cache the hooks we need; caching everything wastes a lot of memory.
    wanted = set(names)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    # Each cache[n] is (B, L, d_model); stacking gives one tensor indexed by depth.
    return torch.stack([cache[n] for n in names]).float()


def run_clause(model, cfg: ExperimentConfig, pool: np.ndarray, clause: str) -> dict:
    """Run every trial for one perturbation clause and return the delta arrays."""
    device = model.cfg.device
    names = resid_hook_names(model.cfg.n_layers)
    # Tokenize the clause WITHOUT a BOS, e.g. " to the store" -> 3 ids.
    clause_ids = model.to_tokens(clause, prepend_bos=False)[0].tolist()
    bos_id = model.tokenizer.bos_token_id
    # Index shift: with BOS at position 0, everything moves right by one.
    off = 1 if cfg.prepend_bos else 0

    # Same seed for every clause -> runs are reproducible clause by clause.
    rng = np.random.default_rng(cfg.seed)

    # Per-batch results, concatenated along the trial axis at the end.
    abs_parts, rel_parts, cos_parts = [], [], []
    max_prefix_leak = 0.0
    n_batches = math.ceil(cfg.n_trials / cfg.batch_size)

    for i in range(n_batches):
        # Last batch may be smaller if n_trials isn't a multiple of batch_size.
        bs = min(cfg.batch_size, cfg.n_trials - i * cfg.batch_size)
        batch = sample_batch(rng, pool, clause_ids, cfg.n_prefix, cfg.n_suffix, bs)

        # Prepend the BOS column to both sequences (same BOS in both runs).
        base, pert = batch.baseline, batch.perturbed
        if cfg.prepend_bos:
            col = np.full((bs, 1), bos_id, dtype=np.int64)
            base, pert = np.hstack([col, base]), np.hstack([col, pert])

        # Two forward passes: fully random baseline and perturbed version.
        # Each result is (13 depths, B, L, 768).
        r_base = cache_residuals(model, torch.from_numpy(base).to(device), names)
        r_pert = cache_residuals(model, torch.from_numpy(pert).to(device), names)

        # Sanity check: causal attention means positions BEFORE the perturbation
        # must be identical in both runs. Anything non-trivial here is a bug.
        pre = slice(0, off + batch.pert_start)
        max_prefix_leak = max(
            max_prefix_leak, (r_pert[:, :, pre] - r_base[:, :, pre]).abs().max().item()
        )

        # Suffix = the 32 tokens AFTER the perturbation. Index j in this slice is
        # distance j+1 from the end of the perturbation.
        suf = slice(off + batch.pert_end, off + batch.pert_end + cfg.n_suffix)
        rb, rp = r_base[:, :, suf], r_pert[:, :, suf]   # (D, B, S, d)
        # Norms are over d_model (the last axis): one number per depth/trial/position.
        d_abs = (rp - rb).norm(dim=-1)                   # (D, B, S)
        d_rel = d_abs / rb.norm(dim=-1).clamp_min(1e-8)
        d_cos = 1 - F.cosine_similarity(rp, rb, dim=-1)

        abs_parts.append(d_abs.cpu().numpy())
        rel_parts.append(d_rel.cpu().numpy())
        cos_parts.append(d_cos.cpu().numpy())

    return {
        "clause": clause,
        "clause_ids": np.array(clause_ids),
        "clause_tokens": [model.tokenizer.decode([t]) for t in clause_ids],
        "abs": np.concatenate(abs_parts, axis=1),
        "rel": np.concatenate(rel_parts, axis=1),
        "cosdist": np.concatenate(cos_parts, axis=1),
        "distances": np.arange(1, cfg.n_suffix + 1),
        "layer_labels": np.array(resid_labels(model.cfg.n_layers)),
        "max_prefix_leak": max_prefix_leak,
    }


def run_experiment(cfg: ExperimentConfig, model=None) -> Path:
    """Run every clause in cfg.perturbations and save results. Returns the run dir."""
    # Seed torch too (numpy sampling is seeded per clause in run_clause).
    torch.manual_seed(cfg.seed)
    run_name = cfg.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save the config FIRST so even a crashed run records what it tried to do.
    cfg.save(run_dir / "config.json")

    t0 = time.time()
    if model is None:
        model = load_model(cfg.model_name, cfg.device)

    log_probs = bos_log_probs(model)
    # Never sample BOS/EOS as a "random" token (both are id 50256 in GPT-2).
    special = {model.tokenizer.bos_token_id, model.tokenizer.eos_token_id}
    pool = build_candidate_pool(
        log_probs, cfg.drop_top_frac, cfg.drop_bottom_frac, exclude_ids=sorted(special)
    )
    print(f"[pool] {len(pool)} candidate tokens out of {len(log_probs)}")

    summary = {"run_name": run_name, "pool_size": int(len(pool)), "clauses": {}}
    for clause in cfg.perturbations:
        tc = time.time()
        res = run_clause(model, cfg, pool, clause)
        slug = slugify(clause)
        # Raw arrays go in a compressed .npz so plots can be redone without the model.
        np.savez_compressed(
            run_dir / f"{slug}.npz",
            **{k: v for k, v in res.items() if k not in ("clause", "max_prefix_leak")},
            clause=np.array(clause),
        )
        summary["clauses"][clause] = {
            "file": f"{slug}.npz",
            "tokens": res["clause_tokens"],
            "n_trials": int(res["abs"].shape[1]),
            "max_prefix_leak": res["max_prefix_leak"],
            "seconds": round(time.time() - tc, 1),
        }
        print(
            f"[{clause!r}] tokens={res['clause_tokens']} "
            f"prefix_leak={res['max_prefix_leak']:.2e} ({time.time() - tc:.1f}s)"
        )

    summary["total_seconds"] = round(time.time() - t0, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
