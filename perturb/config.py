"""Experiment configuration.

Every knob for a run lives here, so a run is fully described by one config.
It is saved as JSON next to its results (results/<run_name>/config.json).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    # --- model ---------------------------------------------------------------
    # gpt2-small, same model as the IOI paper. Loaded through TransformerLens.
    model_name: str = "gpt2-small"
    device: str | None = None  # None -> auto ("cuda" if available, else "cpu")

    # --- sequence layout -----------------------------------------------------
    # [BOS] + [n_prefix random tokens] + [perturbation] + [n_suffix random tokens]
    n_prefix: int = 32
    n_suffix: int = 32
    prepend_bos: bool = True

    # Perturbation clauses. Leading spaces matter for GPT-2's tokenizer:
    # " to the store" -> [" to", " the", " store"] (3 tokens).
    perturbations: list[str] = field(
        default_factory=lambda: [" to the store", " Alice and Bob"]
    )

    # --- random-token pool ---------------------------------------------------
    # Tokens are ranked by the model's next-token probability given only BOS.
    # We drop the most-likely and least-likely ends of that ranking, then sample
    # uniformly (without replacement) from what's left.
    drop_top_frac: float = 0.01      # drop the top 1% most-likely tokens
    drop_bottom_frac: float = 0.25   # drop the bottom 25% (rare / glitch tokens)

    # --- trials --------------------------------------------------------------
    n_trials: int = 200    # random sequence pairs per perturbation
    batch_size: int = 50
    seed: int = 0

    # --- output --------------------------------------------------------------
    output_dir: str = "results"
    run_name: str | None = None  # None -> timestamped name

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text()))
