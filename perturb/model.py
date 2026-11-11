"""Model loading and residual-stream hook names (TransformerLens).

Requires transformer_lens < 4: version 4.0 removed HookedTransformer
(replaced by TransformerBridge). requirements.txt pins this.
"""

from __future__ import annotations

import numpy as np
import torch


def pick_device(device: str | None) -> str:
    """Use the device the user asked for, else GPU if available, else CPU."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = "gpt2-small", device: str | None = None):
    """Load a HookedTransformer. First call downloads weights from HuggingFace (~500MB).

    Weights are cached afterwards (in ~/.cache/huggingface), so later runs are fast.
    from_pretrained applies TransformerLens's default weight processing
    (fold_ln, center_writing_weights, center_unembed). Outputs are unchanged, but the
    residual stream is mean-centered; see docs/experiment_design.md.
    """
    # Imported inside the function so the rest of the package (e.g. the numpy-only
    # sequence code and its tests) works without transformer_lens installed.
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(name, device=pick_device(device))
    model.eval()  # inference only: no dropout
    return model


@torch.no_grad()  # no gradients needed, saves memory
def bos_log_probs(model) -> np.ndarray:
    """Next-token log-probs given only the BOS token: shape (vocab,).

    This is how we rank tokens as "most likely" / "least likely" before building
    the random-token pool. (One of several reasonable choices; see
    docs/experiment_design.md, open question 1.)
    """
    # A batch of one sequence containing just <BOS> (id 50256 for GPT-2).
    bos = torch.tensor([[model.tokenizer.bos_token_id]], device=model.cfg.device)
    logits = model(bos)  # (1, 1, vocab)
    # Logits at the last (only) position -> log-probabilities over the vocab.
    return logits[0, -1].float().log_softmax(-1).cpu().numpy()


def resid_hook_names(n_layers: int) -> list[str]:
    """Residual stream at every depth: before block 0, then after each block.

    For gpt2-small that's 13 points: embed (tok + pos), L0 ... L11.
    resid_pre of block 0 is exactly the token + positional embedding.
    resid_post of block l is the stream after attention AND MLP of layer l.
    """
    return ["blocks.0.hook_resid_pre"] + [
        f"blocks.{l}.hook_resid_post" for l in range(n_layers)
    ]


def resid_labels(n_layers: int) -> list[str]:
    """Human-readable names matching resid_hook_names, used in plots."""
    return ["embed"] + [f"L{l}" for l in range(n_layers)]
