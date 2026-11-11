"""Plots for a finished run. Reads only the saved .npz files, so you can replot
without rerunning the model:  python scripts/run_delta_experiment.py --plot-only results/<run>
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no GUI window
import matplotlib.pyplot as plt
import numpy as np

METRIC_TITLES = {
    "abs": "||Δ residual||  (absolute)",
    "rel": "||Δ|| / ||baseline||  (relative)",
    "cosdist": "1 − cos(perturbed, baseline)",
}


def _mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (D, N, S) -> mean and standard error over trials, each (D, S)."""
    # Standard error of the mean = sample std / sqrt(n trials).
    n = x.shape[1]
    return x.mean(axis=1), x.std(axis=1, ddof=1) / np.sqrt(max(n, 1))


def plot_clause(npz_path: Path, out_dir: Path) -> list[Path]:
    data = np.load(npz_path)
    clause = str(data["clause"])
    labels = [str(l) for l in data["layer_labels"]]
    dist = data["distances"]
    n_trials = data["abs"].shape[1]
    # One color per depth: dark purple = embed, yellow = last layer.
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    out = []

    # 1) Line plot: delta vs distance, one line per depth, for each metric.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["abs", "rel", "cosdist"]):
        mean, sem = _mean_sem(data[metric])
        for d, (lab, c) in enumerate(zip(labels, colors)):
            ax.plot(dist, mean[d], color=c, label=lab, lw=1.6)
            # Shaded band = ±1 standard error over trials.
            ax.fill_between(dist, mean[d] - sem[d], mean[d] + sem[d], color=c, alpha=0.15)
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("distance from end of perturbation (tokens)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean over trials (±1 SE)")
    axes[-1].legend(title="depth", fontsize=8, ncol=2)
    fig.suptitle(f"Perturbation {clause!r} — {n_trials} trials", fontsize=13)
    fig.tight_layout()
    p = out_dir / f"{npz_path.stem}_delta_vs_distance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # 2) Heatmap: depth x distance for the relative metric.
    # Relative metric for the heatmap: raw norms grow with depth and would
    # make the early layers look like zero.
    mean, _ = _mean_sem(data["rel"])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mean, aspect="auto", origin="lower", cmap="magma",
                   extent=[dist[0] - 0.5, dist[-1] + 0.5, -0.5, len(labels) - 0.5])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("distance from end of perturbation (tokens)")
    ax.set_ylabel("residual stream depth")
    ax.set_title(f"Relative Δ, perturbation {clause!r}")
    fig.colorbar(im, ax=ax, label="||Δ|| / ||baseline||")
    fig.tight_layout()
    p = out_dir / f"{npz_path.stem}_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def plot_comparison(npz_paths: list[Path], out_dir: Path, metric: str = "rel") -> Path:
    """All clauses on one figure, one panel per depth (shared y axis)."""
    datas = [np.load(p) for p in npz_paths]
    labels = [str(l) for l in datas[0]["layer_labels"]]
    n = len(labels)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for d in range(n):
        ax = axes[d]
        for data in datas:
            mean, sem = _mean_sem(data[metric])
            ax.plot(data["distances"], mean[d], label=repr(str(data["clause"])))
            ax.fill_between(data["distances"], mean[d] - sem[d], mean[d] + sem[d], alpha=0.15)
        ax.set_title(labels[d])
        ax.grid(alpha=0.3)
    # Hide the unused panels in the last row.
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Clause comparison — {METRIC_TITLES[metric]}", fontsize=13)
    fig.supxlabel("distance from end of perturbation (tokens)")
    fig.tight_layout()
    p = out_dir / f"compare_clauses_{metric}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    npzs = sorted(run_dir.glob("*.npz"))
    out = []
    for p in npzs:
        out += plot_clause(p, run_dir)
    if len(npzs) > 1:
        out.append(plot_comparison(npzs, run_dir))
    return out
