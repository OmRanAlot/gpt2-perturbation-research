"""Run the residual-stream delta experiment and make the plots.

Examples (from the project root):
    python scripts/run_delta_experiment.py                       # defaults
    python scripts/run_delta_experiment.py --n-trials 20         # quick smoke run
    python scripts/run_delta_experiment.py --clauses " to the store" " Alice and Bob" " the cat sat"
    python scripts/run_delta_experiment.py --plot-only results/run_20261105_210000

Outputs:
    results/<run_name>/   raw .npz arrays, config.json, summary.json, all plots
    figures/<run_name>/   copies of the plots with descriptive, dated names
                          (e.g. 2026-11-05_first_pass_to_the_store_delta_vs_distance.png)
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

import _path  # noqa: F401  (adds the project root to sys.path so `perturb` imports)

from perturb.config import ExperimentConfig
from perturb.experiment import run_experiment
from perturb.plotting import plot_run


def export_figures(pngs: list[Path], run_dir: Path, figures_root: str) -> list[Path]:
    """Copy plots into figures/<run_name>/ as <date>_<run_name>_<plot>.png.

    results/ is gitignored (big .npz files); figures/ is the folder of finished
    plots that are safe to commit, share, or attach to an email.
    """
    run_name = run_dir.name
    out_dir = Path(figures_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    out = []
    for p in pngs:
        dst = out_dir / f"{stamp}_{run_name}_{p.name}"
        shutil.copy2(p, dst)
        out.append(dst)
    return out


def main() -> None:
    d = ExperimentConfig()  # defaults come from perturb/config.py
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag below overrides one field of ExperimentConfig.
    ap.add_argument("--clauses", nargs="+", default=d.perturbations,
                    help="perturbation clauses (keep the leading space!)")
    ap.add_argument("--n-trials", type=int, default=d.n_trials)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--n-prefix", type=int, default=d.n_prefix)
    ap.add_argument("--n-suffix", type=int, default=d.n_suffix)
    ap.add_argument("--drop-top", type=float, default=d.drop_top_frac)
    ap.add_argument("--drop-bottom", type=float, default=d.drop_bottom_frac)
    ap.add_argument("--no-bos", action="store_true", help="don't prepend BOS")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default=d.output_dir)
    ap.add_argument("--figures-dir", default="figures",
                    help="where named copies of the plots go (default: figures/)")
    ap.add_argument("--plot-only", metavar="RUN_DIR", help="just replot an existing run")
    args = ap.parse_args()

    # Replot mode: read saved .npz files, no model needed.
    if args.plot_only:
        run_dir = Path(args.plot_only)
        pngs = plot_run(run_dir)
        for p in export_figures(pngs, run_dir, args.figures_dir):
            print("wrote", p)
        return

    cfg = ExperimentConfig(
        perturbations=args.clauses, n_trials=args.n_trials, batch_size=args.batch_size,
        n_prefix=args.n_prefix, n_suffix=args.n_suffix,
        drop_top_frac=args.drop_top, drop_bottom_frac=args.drop_bottom,
        prepend_bos=not args.no_bos, seed=args.seed, device=args.device,
        run_name=args.run_name, output_dir=args.output_dir,
    )
    run_dir = run_experiment(cfg)       # model forward passes + saved arrays
    pngs = plot_run(run_dir)            # plots written next to the arrays
    for p in export_figures(pngs, run_dir, args.figures_dir):
        print("wrote", p)
    print(f"\nDone. Raw results in {run_dir}, figures in {Path(args.figures_dir) / run_dir.name}")


if __name__ == "__main__":
    main()
