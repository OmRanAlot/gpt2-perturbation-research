# Research Log

Newest entries at the top. Keep each entry short: what I did, what I saw, what's next.

---

## 2026-11-05 (later): uv setup + one-shot script

**Done**
- Python pinned to 3.11 (`.python-version`); venv managed with uv.
- `setup_and_run.ps1` creates the venv, installs, runs tests, preview, and the
  first pass, logging to `logs\`.
- Plots are now also copied to `figures/<run_name>/` with dated, descriptive names.
- Added inline comments throughout the code explaining each step.
- Requirements: `torch>=2.6`, `transformer_lens>=3.0,<4`.

**Next**
- Run `setup_and_run.ps1`, check sanity checks in `results\first_pass\summary.json`,
  look at `figures\first_pass\`, write the email to the prof.

---

## 2026-11-05: project setup

**Done**
- Set up the repo: config, sequence generation, experiment runner, plotting, CLI.
- Unit tests for sequence generation (layout, no repeats, shared prefix/suffix,
  reproducibility).
- Wrote down every unspecified decision as an open question
  (`experiment_design.md` §3).

- Verified the sequence tests pass (6/6) and that every API the code uses exists
  in TransformerLens 3.9. Pinned `transformer_lens<4` because 4.0 removed
  `HookedTransformer`.

**Not yet done**
- First real run on gpt2-small. The model code (`experiment.py`, `plotting.py`)
  compiles but hasn't been run end to end yet; it needs the model download on my laptop.

**Next**
1. `setup_and_run.ps1` (does steps 1–4 of the old plan in one go)
5. Check sanity checks (prefix leak ≈ 0, embed line = 0), then look at the plots.
6. Email the prof with the plot + open questions.
