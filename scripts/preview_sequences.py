"""Print a decoded example pair so you can eyeball what the model actually sees.

    python scripts/preview_sequences.py
    python scripts/preview_sequences.py --clause " Alice and Bob" --drop-bottom 0.4
"""

import argparse

import _path  # noqa: F401
import numpy as np

from perturb.config import ExperimentConfig
from perturb.model import bos_log_probs, load_model
from perturb.sequences import build_candidate_pool, sample_batch


def main() -> None:
    d = ExperimentConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--clause", default=d.perturbations[0])
    ap.add_argument("--drop-top", type=float, default=d.drop_top_frac)
    ap.add_argument("--drop-bottom", type=float, default=d.drop_bottom_frac)
    ap.add_argument("--seed", type=int, default=d.seed)
    args = ap.parse_args()

    model = load_model(d.model_name)
    tok = model.tokenizer
    lp = bos_log_probs(model)
    pool = build_candidate_pool(lp, args.drop_top, args.drop_bottom,
                                exclude_ids=[tok.bos_token_id, tok.eos_token_id])
    order = np.argsort(lp)
    show = lambda ids: [tok.decode([int(i)]) for i in ids]
    print(f"vocab={len(lp)}  pool={len(pool)}")
    print("most likely (dropped):  ", show(order[::-1][:15]))
    print("least likely (dropped): ", show(order[:15]))

    clause_ids = model.to_tokens(args.clause, prepend_bos=False)[0].tolist()
    print(f"\nclause {args.clause!r} -> {show(clause_ids)}")
    b = sample_batch(np.random.default_rng(args.seed), pool, clause_ids, d.n_prefix, d.n_suffix, 1)
    print("\nBASELINE:\n", repr(tok.decode(b.baseline[0])))
    print("\nPERTURBED:\n", repr(tok.decode(b.perturbed[0])))


if __name__ == "__main__":
    main()
