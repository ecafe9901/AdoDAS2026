# ADODAS2026 Code Review

Review date: 2026-05-24
Based on commit `79dbbb3` (V2: remove focal loss, 6-layer TCN, split LLM features, NaN guard).

This document analyzes two sets of experimental results:
- **V2** — from `results/v2_training_results.md` (commit `79dbbb3`), using CORALHead, γ=0.0, 13-dim behavioral LLM features
- **V1-MTL** — a separate experimental version with 5-task MTL loss (CE + BCE + MSE + CLL + CONS), not in git history

---

## 1. V2 Results Analysis

### Config

| Param | V2 | V1 baseline |
|-------|----|-------------|
| tcn_layers | 6 | 4 |
| gamma (focal loss) | 0.0 | 2.0 |
| LLM dims (A2) | 13 (behavioral only) | 34 (all) |
| batch_size | 24 | 32 |
| early_stop_metric | primary (QWK) | val_loss |
| patience | 10 | 8 |
| head | CORAL | CORAL |

### Training Results

| Epoch | Train Loss | Val Loss | QWK | 0% | 1% | 2% | 3% |
|-------|-----------|----------|-----|----|----|----|----|
| 1 | 1.44 | 0.637 | 0.001 | 28 | 72 | 0 | 0 |
| **7** | **1.20** | **0.611** | **0.070** | 28 | 72 | 0 | 0 |
| 17 | 1.19 | 0.611 | 0.067 | 27 | 73 | 0 | 0 |

GT distribution: 0=69.4%, 1=23.4%, 2=4.7%, 3=2.5%

### Calibration Results

| Strategy | QWK | 0% | 1% | 2% | 3% |
|----------|-----|----|----|----|----|
| argmax (raw) | 0.007 | 95 | 5 | 0 | 0 |
| expectation (raw) | 0.070 | 28 | 72 | 0 | 0 |
| **calibrated_argmax** | **0.096** | **46** | **52** | **2** | **0** |
| calibrated_expectation | 0.096 | 46 | 55 | 0 | 0 |

### Observation: 0→1 Distribution Flip

V2's core failure: pos_weight pushes every sample's CORAL scores uniformly upward, causing the model to predict 1 instead of 0 for most samples, but never reaching 2 or 3.

The per-threshold pos_weight:
- Threshold 1 (≥1): sqrt(70/30) ≈ 1.53
- Threshold 2 (≥2): sqrt(90.5/9.5) ≈ 3.09
- Threshold 3 (≥3): sqrt(97.5/2.5) ≈ 6.24

Because CORALHead shares a single `score_fc` across all 3 thresholds, the gradient from threshold 3's 6× weight flows back through the same parameters as threshold 1's 1.5× weight. The net effect is a uniform right-shift of all scores rather than threshold-specific adaptation.

### Observation: Training Plateaus Early

Train loss drops from 1.44 (epoch 1) to 1.20 (epoch 7), then flatlines at ~1.19. Cosine LR decays from 1e-3 to 2.9e-4 by epoch 17. The model stops learning useful representations after epoch ~7 and merely adjusts the 0/1 decision boundary.

### V2 Summary

| Issue | Evidence | Impact |
|-------|----------|--------|
| pos_weight over-shifts all scores | Pred dist 28/72/0/0 vs GT 69/23/5/3 | Model never predicts 2 or 3 |
| CORALHead shared score_fc | All thresholds share one linear layer | Can't specialize per threshold |
| Cosine LR too aggressive | LR 1e-3→2.9e-4 in 17 epochs | Learning stops at epoch 7 |
| Scores 2-3 never learned | 0% prediction throughout | Without being predicted, can't be calibrated |
| Calibration helps marginally | Raw QWK 0.070→calibrated 0.096 | Compensates bias but not missing classes |

---

## 2. V1-MTL Experiment Analysis

A separate experimental run (not in git history) with 5 concurrent A2 losses:

| Loss | Value | Weight | Role |
|------|-------|--------|------|
| CE (4-Class) | 2.365 | × 1.0 | Treats 0-3 as flat classification |
| BCE (Step) | 0.261 | × 0.5 | Ordinal step-wise BCE |
| MSE (QWK-Exp) | 0.215 | × 2.0 | QWK expectation regression |
| CLL (Mono) | 0.000 | × 1.0 | Monotonicity constraint |
| CONS (KL-Align) | 1.002 | × 0.2 | Distribution consistency |

### Results

- QWK = 0.374, MAE = 0.565, Score Bias = 0.144
- Pred dist: 0=81.8%, 1=0%, 2=0%, 3=18.2%
- CLL = 0.00 (monotonicity constraint not firing)

### Different Collapse Mode

Unlike V2's 0→1 shift, V1-MTL collapses to extremes (0 or 3). CE loss (×1.0) dominates BCE (×0.5), encouraging the model to classify rather than order. The model learns "is this score 0 or 3?" and never assigns intermediate values.

### The CLL=0.00 Anomaly

CLL being exactly 0.00 across all batches is suspicious. Two possible explanations:
1. The implementation only checks CORALHead's `raw_thresholds` parameters (which are guaranteed monotonic by `softplus + cumsum`), not the actual output distributions
2. A bug in the loss calculation causes it to always return zero

This needs code inspection of the CLL implementation.

### Why V1-MTL QWK (0.374) Appears Higher Than V2 (0.070)

V1-MTL's higher QWK is largely an artifact:
- QWK rewards agreement on the extremes more than middle categories
- Predicting 0 for all real-0 and 3 for all real-3 yields high QWK despite never predicting 1 or 2
- Both versions fail to learn ordinal structure; they just fail in different ways

---

## 3. Current Code Issues

### LLM Feature Integration

```
clean_transcript.txt → DeepSeek API → 34-74 dim feature vector → .npy file
                                                                       ↓
                                                              llm_proj: Linear(34→64) + GELU + Linear(64→64)
                                                                       ↓
                                                              concat with participant_repr
```

**A2 offset design**: `llm_offset=21, d_llm=13` correctly skips the 21 DASS items and uses only behavioral markers. This prevents the "predict questionnaire from LLM-extracted questionnaire scores" shortcut.

**Calibration pipeline**: Linear regression per item on training labels, then applied to test. Sound approach.

**V2/V3 divergence**: Multiple LLM feature versions exist (V1=34, V2=41, V3=74 dims). No automated dimension check at load time, creating config drift risk.

### `use_coral` Configuration Path

The V2 results file states `head: CORAL`. The config YAML has `use_coral: true`. The runner code at line ~953 may or may not correctly propagate this — need to verify whether V2 actually ran with CORALHead or silently fell back to A2OrdinalHead.

If V2 actually used A2OrdinalHead (fixed thresholds, non-learnable), the analysis of CORALHead's shared `score_fc` limitation would not apply to V2. This needs clarification.

### Other Code Quality Notes

- **NaN gradient guard**: Works correctly — V2 had zero NaN events over 17 epochs.
- **Length-bucketed batching**: Reduces padding waste from ~72% to ~20%.
- **A2 decode strategy selection**: Auto-selects among 6 strategies (raw/calibrated × 3 decode methods). Works correctly.
- **Joint mode A1 submission bug**: `"a1_preds" in dir()` always returns False.

---

## 4. Recommended Fixes (for V3)

### P0: Training Dynamics

| Fix | Rationale |
|-----|-----------|
| Remove or reduce pos_weight | Stops the uniform 0→1 score shift in V2 |
| Replace cosine LR with ReduceLROnPlateau | Prevents LR from dying before tail classes can emerge |
| Oversample score=2 and score=3 training samples | Gives tail classes direct gradient signal |

### P1: Architecture

| Fix | Rationale |
|-----|-----------|
| Per-threshold bias in CORALHead | Let each threshold adapt independently instead of sharing one score_fc |
| Verify CORALHead is actually being used | Fix `use_coral` config propagation if broken |

### P2: LLM Features

| Fix | Rationale |
|-----|-----------|
| Add dimension assertion at load time | Catch V1/V2/V3 format mismatches |
| Ablate LLM features vs no LLM | Measure actual contribution once ordinal collapse is fixed |

---

## 5. Remaining Known Issues (As of 2026-05-28, commit `8bf1edb`)

Four issues identified from the current codebase. Prioritized by actual impact.

### P3 (No Impact): `"a1_preds" in dir()` — Dead Code

**Location**: [`common/runner.py:1501`](../common/runner.py#L1501)

```python
if joint and "a1_preds" in dir() and len(a1_preds) > 0:
```

`dir()` returns a list of names in the current scope — `"a1_preds" in dir()` always returns `True` in CPython for any local variable assigned anywhere in the function, because the compiler statically allocates all local variable slots. This check never prevents an `UnboundLocalError`.

However, this is dead rather than buggy: `joint=True` and `a1_head is not None` (lines 1427, 1076) always co-occur, so `a1_preds` is always defined when this branch is reached. The `dir()` guard is purely redundant.

**Fix**: Remove the `"a1_preds" in dir()` condition entirely.

---

### P0 (Core Bottleneck): CORALHead Shared `score_fc` Limits Per-Threshold Adaptation

**Location**: [`common/models/grouped_model.py:224-236`](../common/models/grouped_model.py#L224-L236)

All 3 ordinal thresholds share a single `score_fc` linear layer:
```python
logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0)
```

**Evidence from Baseline+ training log (QWK=0.253)**:
- Score=0/1/2 distributions converge to match GT by epoch 10
- Score=3 first appears at epoch 31, matches GT proportion (2.4%) by epoch 34
- But QWK **drops** from 0.234 → 0.211 when score=3 emerges

**Root cause**: Raising `scores` to cross threshold 3 also pushes logits for thresholds 1 and 2 upward, disturbing already-learned boundaries. The model eventually produces the right *number* of score=3 predictions but cannot *discriminate which samples deserve them*.

**Fix**: Add a per-threshold bias to CORALHead:
```python
self.threshold_bias = nn.Parameter(torch.zeros(1, n_items, n_thresholds))
# Forward:
logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0) + self.threshold_bias
```
Initialize bias at 0 and use a lower learning rate (e.g. via separate param group with `lr_mult=0.1`) to avoid disturbing thresholds 1-2.

---

### P1 (Secondary): Cosine LR `eta_min=1e-6` Too Aggressive

**Location**: [`common/runner.py:206`](../common/runner.py#L206)

```python
cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
```

By epoch 30 (out of 40), LR decays to ~1.1e-4. By epoch 34 (when score=3 first appears at correct proportion), LR is 6.4e-5 — too low for meaningful boundary adjustment.

**Not the primary bottleneck**: score=3's precision problem persists even at higher LR (epochs 31-33). The model needs architectural change (per-threshold bias) more than LR schedule change.

**Recommended fix**: Either raise `eta_min` to 5e-5, or switch to `ReduceLROnPlateau(factor=0.5, patience=5)` with a floor of 1e-5.

---

### P2 (Minor): Uniform label_smoothing Across All Classes

**Location**: [`common/models/heads.py:92-93`](../common/models/heads.py#L92)

```python
if label_smoothing > 0.0:
    targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
```

**Analysis**: With `label_smoothing=0.05`, score=3 targets (2.5% of samples) are diluted from 1.0 → 0.975, a 2.5% gradient reduction. The BCE gradient `sigmoid(logit) - target` shifts by only 0.025. Compared to pos_weight=6.24 amplifying threshold 3 gradients 6×, this smoothing effect is negligible.

**Evidence**: Score=2 (4.7% of samples, same pos_weight=3.09) converges correctly from epoch 7 onward. Label smoothing is not preventing tail learning.

**Recommended fix**: Lower to 0.02 for safety. Per-threshold smoothing is unnecessary complexity for negligible gain.
