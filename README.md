# GPT-2 Residual-Stream Perturbation Experiment

Research project with my NJIT ML professor. Builds on induction-head work.

**Question:** if you drop a short meaningful clause (e.g. `" to the store"`) into
the middle of a sequence of random tokens, how far does that "disturbance"
travel downstream through gpt2-small's residual stream, and how does that
change with depth?

**First analysis:** magnitude of the residual-stream difference (perturbed vs.
fully random baseline) at every layer, plotted against distance from the end of
the perturbation.

Full spec, every implementation decision, and open questions for the professor:
[`docs/experiment_design.md`](docs/experiment_design.md).
Progress notes: [`docs/research_log.md`](docs/research_log.md).

---

## Setup (Windows / PowerShell, uv)

**One command does everything** (venv, install, tests, preview, first pass, figures):

```powershell
cd ~\code\gpt2-perturbation-research
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
```

Needs [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv` or `pip install uv`).
Python is pinned to **3.11** in `.python-version`; uv downloads it if missing.
The output is also saved to `logs\`.

Manual version of the same thing:

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

The first model load downloads gpt2-small from HuggingFace (~500 MB, cached after).
gpt2-small runs fine on CPU; the default run (2 clauses × 200 trials) takes a few minutes.

## Run it

```powershell
# 1. Unit tests for sequence generation (no model needed, ~1s)
python -m pytest tests

# 2. Eyeball what the model actually sees: pool stats + a decoded example pair
python scripts\preview_sequences.py

# 3. Quick smoke run
python scripts\run_delta_experiment.py --n-trials 20 --run-name smoke

# 4. Real run
python scripts\run_delta_experiment.py --run-name first_pass

# More clauses (keep the leading space!)
python scripts\run_delta_experiment.py --clauses " to the store" " Alice and Bob" " the red car" --run-name three_clauses

# Re-plot an old run without rerunning the model
python scripts\run_delta_experiment.py --plot-only results\first_pass
```

All CLI flags: `python scripts\run_delta_experiment.py --help`. Defaults live in
`perturb/config.py`.

## What a run produces

**`figures/<run_name>/`**: the finished plots, named `<date>_<run_name>_<plot>.png`,
e.g. `2026-11-05_first_pass_to_the_store_delta_vs_distance.png`. This is the
folder to look at / send to the prof (committed to git).

**`results/<run_name>/`**: everything raw (gitignored):

| file | what it is |
|---|---|
| `config.json` | exact config used, so every run is reproducible |
| `summary.json` | pool size, clause tokenization, timing, **prefix-leak sanity check** |
| `<clause>.npz` | raw arrays `abs`, `rel`, `cosdist`, each shaped `(13 depths, n_trials, 32 distances)` |
| `<clause>_delta_vs_distance.png` | **the main plot**: Δ vs distance, one line per depth, 3 metrics |
| `<clause>_heatmap.png` | depth × distance heatmap of relative Δ |
| `compare_clauses_rel.png` | every clause on one figure, one panel per depth |

**Sanity checks to look at first**
- `max_prefix_leak` in `summary.json` should be ~0 (float noise, ~1e-5 or less).
  Causal attention means tokens *before* the perturbation can't see it. If it's
  big, something is wrong.
- The `embed` line should be flat at exactly 0. At suffix positions the token
  and position are identical in both runs, so the difference can only appear
  once attention mixes information in (L0 onward).

## Layout

```
perturb/
  config.py       all knobs (ExperimentConfig dataclass)
  sequences.py    random-token pool + baseline/perturbed pair sampling (numpy only)
  model.py        load gpt2-small via TransformerLens, BOS log-probs, hook names
  experiment.py   run both sequences, cache residuals, compute deltas, save
  plotting.py     plots from saved .npz files
scripts/
  run_delta_experiment.py   main entry point (CLI)
  preview_sequences.py      print a decoded example pair
tests/
  test_sequences.py         unit tests for sequence generation
docs/
  experiment_design.md      spec, decisions, open questions for the prof
  research_log.md           dated progress notes
results/                    raw run outputs (gitignored)
figures/                    named plot copies per run (committed)
logs/                       setup_and_run.ps1 logs (gitignored)
setup_and_run.ps1           one-shot uv setup + tests + first pass
.python-version             3.11 (read by uv)
```

## Stack

Python 3.11 (uv-managed venv), PyTorch, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
(hooks into every residual-stream point via `run_with_cache`), NumPy, Matplotlib, pytest.
