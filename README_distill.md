# GazeLLE Distillation Guide

This note summarizes how to train a smaller GazeLLE variant (e.g. `gazelle_dinov3_vit_tiny`) under a teacher such as `gazelle_dinov3_vits16plus` using the distillation hooks wired into the training scripts.

## 1. Overview
- Distillation is disabled by default. It becomes active only when `--distill_weight` is set to a positive value.
- Both `scripts/train_gazefollow.py` and `scripts/train_vat.py` now accept the same pair of options:
  - `--distill_teacher`: teacher model name (defaults to `gazelle_dinov3_vits16plus`).
  - `--distill_weight`: scalar weight applied to the auxiliary MSE loss between teacher and student heatmaps.
- When enabled, the teacher model is loaded in evaluation mode, frozen, and kept in `torch.no_grad()` contexts during training.

## 2. Enabling Distillation
```bash
# Example: distilling the ViT-Tiny student on GazeFollow
uv run python scripts/train_gazefollow.py \
--model_name gazelle_dinov3_vit_tiny \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3
```

```bash
# Example: distilling the in/out model on VAT
uv run python scripts/train_vat.py \
--model gazelle_dinov3_vit_tinyplus \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3
```

Passing `--distill_weight 0` (or omitting the flag) keeps the previous training behaviour.

## 3. Teacher Choice
- Start with `gazelle_dinov3_vits16plus`: it offers a strong signal while remaining light enough to co-train with the student.
- Once the pipeline is stable, consider swapping to `gazelle_dinov3_vitb16` for potential accuracy gains. Expect to revisit loss weights because the representational gap grows with the larger teacher.
- If the student and teacher share the same architecture, expect marginal benefit; the scripts still allow this configuration but emit a warning.

## 4. Loss Formulation
- Student heatmaps use BCE/BCEWithLogits (identical to the original training setup).
- The distillation term aligns student and teacher heatmaps via MSE:
  - `train_gazefollow.py`: if sigmoid is disabled, logits are converted to probabilities before computing the MSE.
  - `train_vat.py`: teacher/student probabilities are compared directly (both already in `[0, 1]`).
- The total loss becomes `L_total = L_supervised + distill_weight * L_distill`.

## 5. Picking `distill_weight`
- Begin around **0.3** — this keeps the gradient scale comparable to the BCE term during early training.
- Run a short sweep (`0.1`, `0.3`, `0.5`, `1.0`) and watch:
  - If BCE fails to decrease or explodes, the weight is too high.
  - If teacher/student curves barely couple (distill loss remains large), the weight is too small.
- Warm-up trick: start at `0` and linearly ramp to the target value over the first epoch if training becomes unstable.

## 6. Logging & Monitoring
- New TensorBoard tags:
  - `train/distill_loss` is logged when distillation is active (both scripts).
- Keep tracking the existing metrics (AUC, L2, AP). A healthy run will show the distill loss trending down while the supervised metrics continue improving.

## 7. Checkpoint & Resume Behaviour
- The new CLI flags are stored inside checkpoints. Resuming a run automatically restores `--distill_weight` and `--distill_teacher` to their saved values and prints a warning if the CLI input differs.
- Teacher weights are **not** saved in checkpoints; reloading the teacher is deterministic because it is instantiated from the provided model name and default checkpoints (`./ckpts/...`).

## 8. Practical Tips
- **GPU memory**: running both student and teacher forward passes roughly doubles memory usage. Use AMP (`--use_amp`) or smaller batch sizes if you encounter OOM errors.
- **Data augmentations**: strong augmentations help the student generalize when imitating a larger teacher. Ensure teacher and student receive the *same* inputs to keep the supervision consistent.
- **Sanity checks**:
  - Set `--distill_weight` to a very large value (e.g. `10`) for a few iterations; the student heatmaps should quickly mimic the teacher. Revert afterwards.
  - Run one training step with `--distill_weight 0` and confirm the loss matches the historical baseline.
- **Teacher checkpoints**: verify the teacher’s base performance before distilling. Garbage in leads to garbage out.

## 9. Next Steps
1. Launch a short training run with `--distill_weight 0.3` to confirm the pipeline works end-to-end.
2. Evaluate the student against the previous non-distilled checkpoint to quantify gains.
3. Explore mid-level feature or attention-map matching if the student accuracy plateaus.

With these pieces in place, you can iterate on teacher choices, loss weighting, and additional hints to tailor the distillation strategy to your needs.
