# Experiment Design

## 1. What the professor asked for (from our meeting, ~late Sept 2026)

- Builds on induction work.
- Random token sequences: drop the **most likely and least likely** tokens, then
  **sample uniformly** from what's left, with **no repeated tokens**.
- Layout: **32 random tokens → perturbation → 32 random tokens**.
- Perturbations start as **3-word clauses**, e.g. "to the store", "Alice and Bob".
- Compare the **fully random** sequence against the **perturbed** one, looking at
  the **second** random run (the 32 tokens after the perturbation).
- First analysis: **magnitude of the delta at different residual-stream depths**,
  plotted against **distance from the end of the perturbation**.
- Model: **gpt2-small** (same as the IOI paper), from HuggingFace.

## 2. How it's implemented

### Sequences (`perturb/sequences.py`)

```
            position:  0     1 .. 32          33 .. 35         36 .. 67
baseline : [BOS] [prefix: 32 random] [filler: 3 random] [suffix: 32 random]
perturbed: [BOS] [prefix: 32 random] [clause: 3 tokens] [suffix: 32 random]
```

- Prefix and suffix are **identical** in the two runs. Only the middle slot differs.
- **Distance** = suffix index + 1, so the first token after the perturbation is
  distance 1 and the last is distance 32.
- **Within a sequence, no token repeats.** Prefix, filler and suffix are one
  draw without replacement, and the clause's tokens are removed from the pool.
  This matters for an induction-adjacent experiment: a repeated token would let
  induction heads fire and confound the result.

### Token pool (`build_candidate_pool`)

1. Get the model's next-token distribution given only `<BOS>`.
2. Sort all 50,257 tokens by that probability.
3. Drop the top `drop_top_frac` (default 1%) and bottom `drop_bottom_frac`
   (default 25%). The bottom end is mostly junk / under-trained "glitch" tokens.
4. Remove BOS/EOS (both id 50256 in GPT-2).
5. Sample uniformly from what remains.

### Measurement (`perturb/experiment.py`)

Residual stream at 13 depths: `embed` (= `blocks.0.hook_resid_pre`, token +
position embedding) and `L0..L11` (= `blocks.{l}.hook_resid_post`).

At each depth and suffix position, with `r_b` = baseline residual, `r_p` = perturbed:

| metric | formula | why |
|---|---|---|
| `abs` | ‖r_p − r_b‖₂ | what the prof asked for |
| `rel` | ‖r_p − r_b‖₂ / ‖r_b‖₂ | residual norm grows a lot with depth in GPT-2, so raw `abs` mostly shows "later layers are bigger"; `rel` makes layers comparable |
| `cosdist` | 1 − cos(r_p, r_b) | direction change, ignores scale |

Averaged over `n_trials` (default 200) random pairs; plots show mean ± 1 standard error.

### Model loading note

`HookedTransformer.from_pretrained("gpt2-small")` applies TransformerLens's default
weight processing (`fold_ln`, `center_writing_weights`, `center_unembed`). That
doesn't change the model's outputs, but the residual stream is mean-centered, so
absolute norms differ a bit from raw HuggingFace GPT-2. Standard in interp work;
mention it if the prof compares numbers with another codebase.

TransformerLens **4.0 removed `HookedTransformer`** (replaced by `TransformerBridge`),
so `requirements.txt` pins `<4`.

### Built-in sanity checks

- **Prefix leak:** positions before the perturbation must match exactly
  (causal masking). Logged as `max_prefix_leak` in `summary.json`.
- **Embed line = 0:** suffix tokens and positions are identical in both runs,
  so the `embed` depth must show zero delta.

## 3. Decisions I made that the prof didn't specify (check with him)

These are the open questions to bring to the next meeting.

1. **What does "most/least likely" mean?** I used the model's next-token
   distribution given only BOS. Alternatives: average distribution over many
   contexts, or unigram frequency from a corpus (e.g. OpenWebText).
2. **How many to drop?** Defaulted to top 1% / bottom 25%. Easy to change
   (`--drop-top`, `--drop-bottom`).
3. **What's in the middle slot of the "fully random" sequence?** I filled it with
   3 random tokens so both sequences have the same length and the suffix sits at
   the same absolute positions. The alternative is to have no middle slot at all
   (64-token baseline), which shifts positions by 3 and mixes a positional effect
   into the delta.
4. **Prepend BOS?** Yes by default (GPT-2 uses position 0 as an attention sink;
   without BOS the first random token takes that role). `--no-bos` to turn off.
5. **Which norm/metric does he want?** I compute absolute, relative and cosine.
   Ask which he cares about.
6. **Leading space on clauses.** `" to the store"` tokenizes to 3 tokens
   (`" to"`, `" the"`, `" store"`). Without the space, `"to"` is a different token.
7. **Next steps after the first plot?** Possible directions: per-head attribution
   (which heads carry the perturbation forward), attention-only vs MLP
   contributions (`hook_attn_out` / `hook_mlp_out`), longer/shorter clauses,
   semantically related vs unrelated suffix tokens.

## 4. Things to watch for in the results

- Does the delta decay with distance, or stay roughly flat? (Flat would suggest
  the clause gets "written" into a persistent direction that attention keeps
  copying forward.)
- Is there a sharp change at a specific layer? Induction / previous-token heads
  in gpt2-small live roughly in L1–L6.
- Distance 1 (the token right after the clause) is probably special: previous-
  token heads look directly at the last clause token.
- Do the two clauses behave differently (" Alice and Bob" has names, which IOI
  circuits care about)?
