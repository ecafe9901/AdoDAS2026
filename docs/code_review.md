# ADODAS2026 Code Review

Review date: 2026-05-24
Based on commit `79dbbb3` (ecafe9901 fork).
V2 training results from `results/v2_training_results.md`.

---

## 1. Critical: A2 Ordinal Regression Collapse

### Symptom: Two Failure Modes

The architecture oscillates between two distinct collapse patterns depending on config:

**V1 (MTL: CE + focal loss γ=2.0, 34-dim LLM):**
```
        Pred 0  Pred 1  Pred 2  Pred 3       GT%
True 0   8045      0       0      697        69.4%
True 1   1948      0       0     1005        23.4%
True 2    223      0       0      364         4.7%
True 3     86      0       0      232         2.5%
```
QWK = 0.374. Pred dist: 0=82%, 1=0%, 2=0%, 3=18%.
**Collapse to extremes (0 or 3).**

**V2 (CORAL only, γ=0.0, 13-dim behavioral LLM):**
```
Epoch 1:  28% 0, 72% 1,  0% 2,  0% 3   QWK=0.001
Epoch 7:  28% 0, 72% 1,  0% 2,  0% 3   QWK=0.070  (best)
Epoch 17: 27% 0, 73% 1,  0% 2,  0% 3   QWK=0.067
```
**Collapse to middle (0 or 1 only).** Score 2-3 never predicted during training.

### Both modes share a common root: CORALHead fails when tail classes are sparse

CORALHead's architecture:
```python
scores = self.score_fc(x)               # (B, 21) — single shared score per item
thresholds = cumsum(softplus(raw))      # (21, 3) — guaranteed monotonic
logits = scores.unsqueeze(-1) - thresholds  # broadcast
```

Key constraint: all 3 thresholds share one `scores` value per item. The model cannot independently learn "this sample has score=2 vs 3" — it must push the entire score higher while keeping thresholds apart.

### V1 Collapse: CE + focal loss pushes to extremes

The MTL setup with 5 concurrent losses:

| Loss | Weight | Effect |
|------|--------|--------|
| CE (4-Class) | × 1.0 | Flat 4-way classification — gives shortcut: "guess 0 or 3" |
| BCE (Step) | × 0.5 | Proper ordinal loss — but half the weight of CE |
| MSE (QWK-Exp) | × 2.0 | Regression target |
| CLL (Mono) | × 1.0 | **CLL = 0.0** — monotonicity constraint not firing |
| CONS (KL-Align) | × 0.2 | Distribution consistency |

CE × 1.0 dominates BCE × 0.5. The model learns "classify as 0 or 3" because flat CE has stronger gradients for polarizing predictions than BCE has for maintaining ordinal structure. CLL = 0.0 suggests the monotonicity check either has a bug or only verifies CORALHead parameter structure (which is already guaranteed monotonic by `softplus + cumsum`) rather than actual output distributions.

**Diagnosis: Loss weight imbalance + CLL dead code.**

### V2 Collapse: pos_weight causes 0→1 flip

The per-threshold `compute_a2_pos_weight`:

```
Threshold 1 (≥1 vs 0):  sqrt(70/30)   = 1.53
Threshold 2 (≥2 vs <2):  sqrt(90.5/9.5)= 3.09  
Threshold 3 (≥3 vs <3):  sqrt(97.5/2.5)= 6.24
```

All 3 thresholds share the same `scores = score_fc(x)`. pos_weight pushes `scores` upward uniformly — the gradient from threshold 3's 6× weight flows back through the **same** `score_fc` weights as threshold 1's 1.5× weight. The result:

```
Before pos_weight:  P(≥1)=0.1  P(≥2)=0.05  P(≥3)=0.02   →  argmax=0
After  pos_weight:  P(≥1)=0.8  P(≥2)=0.6   P(≥3)=0.4    →  argmax=1
```

Every sample shifts from 0→1. The model never predicts 2-3 because the shared score doesn't allow threshold-specific adaptation. The tail thresholds' gradients are diluted by the dominant head threshold's signal.

V2 results confirm: after epoch 4-7, the model stops learning useful representations and just adjusts the 0/1 threshold:

```
Epoch 1:  loss=1.44  Val loss=0.637  QWK=0.001
Epoch 7:  loss=1.20  Val loss=0.611  QWK=0.070  (best, but plateauing)
Epoch 17: loss=1.19  Val loss=0.611  QWK=0.067
```

Training loss flatlined at ~1.19 from epoch 7 onward. Cosine LR decay from 1e-3 to 2.9e-4 by epoch 17 killed learning before any tail class signal could emerge.

### Calibration as Band-Aid

Post-hoc threshold calibration produced the only score=2 predictions in V2:

| Strategy | QWK | 0% | 1% | 2% | 3% |
|----------|-----|----|----|----|----|
| raw expectation | 0.070 | 28 | 72 | 0 | 0 |
| **calibrated_argmax** | **0.096** | **46** | **52** | **2** | **0** |
| calibrated_expectation | 0.096 | 46 | 55 | 0 | 0 |

Calibrated_argmax achieves 2.1% score=2 — first time ever — but QWK is still only 0.096, barely above random. Calibration compensates for systematic bias but cannot create ordinal understanding the model never learned.

### Summary: Two Collapse Modes

| Config | Collapse | QWK | Root Cause |
|--------|----------|-----|------------|
| **V1** (CE+BCE+MTL, γ=2.0) | 0-or-3 | 0.374 | CE dominates BCE; CLL=0 |
| **V2** (CORAL, γ=0.0, 13-dim LLM) | 0-or-1 | 0.070 | pos_weight over-shifts all scores; shared score_fc prevents threshold specialization |
| Ideal | 0/1/2/3 | >0.5 | — |

The "better" V1 QWK (0.374) is misleading — it's an artifact of predicting only extremes on a heavily 0/3 distribution. Neither mode learns genuine ordinal structure.

### Fix Strategy

1. **Remove pos_weight** — stops the uniform 0→1 shift
2. **Oversample score=2 and score=3 samples** in training — gives tail classes direct gradient signal without distorting all thresholds
3. **Replace Cosine LR with ReduceLROnPlateau** — prevents LR dying before tail classes can be learned
4. **Independent score scaling per threshold** — modify CORALHead to allow threshold-specific score offsets: `logits = scores.unsqueeze(-1) - thresholds + per_threshold_bias`

---

## 2. LLM Features: Integration Analysis

### Architecture

```
clean_transcript.txt  →  DeepSeek API  →  34+ dim feature vector
                                            ↓
                                      numpy .npy file
                                            ↓
                                   GroupedParticipantDataset
                                   loads by participant ID
                                            ↓
                                   llm_proj: Linear(34→64) + GELU + Linear(64→64)
                                            ↓
                                   concat with participant_repr
```

### Positive Findings

| Aspect | Assessment |
|--------|-----------|
| **A2 offset design** (`llm_offset=21, d_llm=13`) | Correct — skips DASS items, uses only behavioral markers. Prevents shortcut learning. |
| **Calibration pipeline** | Well-designed: linear regression per item on training labels, then applied to test |
| **Per-session V3 features** (74-dim) | Cross-session inconsistency + emotional range features are well-motivated for depression assessment |
| **Quality gating** | V2 added `quality_score` gating for ASR noise — good defensive design |
| **V2 ablation confirmed correct** | Removing full DASS scores from LLM features (only 13 behavioral dims) prevents the "predict questionnaire from questionnaire-derived features" shortcut |

### Risks

1. **DeepSeek API dependency**: Extraction requires `DEEPSEEK_API_KEY`. Reproducibility depends on API availability and model version pinning.

2. **LLM calibration overfit risk**: Linear regression per item on training data (~1K participants) with 2 params per item = 42 params. With only 2.5% score=3, calibration coefficients for tail classes may be noise-driven.

3. **V1 vs V2/V3 divergence**: Multiple feature versions with different dims (34 / 41 / 74). `llm_feature_dim: 34` in config must exactly match V1 format. No automated dimension check at load time.

4. **Bottleneck width**: 64-dim projection is narrow for 74-dim V3 features. But LLM features only contribute 64/320 = 20% of final representation — this may limit their impact.

5. **LLM features can't fix ordinal collapse**: V2 results show that even with LLM features, the ordinal regression still collapses. The bottleneck is in CORALHead's architecture and training dynamics, not the input features.

### Recommendations

- **Add feature dimension assertion** at dataset load time to catch version mismatches
- **Try wider bottleneck** (128) for V3 74-dim features
- **Run LLM feature ablation**: compare A2 QWK with vs without LLM features once the ordinal collapse is fixed

---

## 3. Config Observations (tasks/a2/default.yaml)

### V2 Config

| Setting | Value | Assessment |
|---------|-------|------------|
| `tcn_layers: 6` | 6 (vs 4) | Good — 1.24s → 2.52s receptive field |
| `weight_decay: 0.02` | Reduced from 0.05 | Correct — 0.05 was too aggressive |
| `session_loss_weight: 1.0` | Up from 0.2 | Reasonable; A01 excluded in code |
| `gamma: 0.0` | Focal loss off | Correct — focal loss + pos_weight overcorrects |
| `label_smoothing: 0.0` | Off | Correct for rare classes |
| `feature_noise_std: 0.01` | Minimal | Safe baseline |
| `session_drop_prob: 0.05` | Down from 0.1 | Conservative dropout |
| `batch_size: 24` | Reduced | OK (limited by GPU memory) |
| `early_stop_metric: primary` | QWK-based | Correct — val_loss plateaus early |
| `patience: 10` | High | Good for slow convergence in tail classes |
| `warmup_epochs: 3` | Standard | OK |
| **`lr: 0.001` + cosine** | Problematic | Cosine decay reaches 2.9e-4 by epoch 17; learning stops before tail classes can emerge |

### body_pose + global_motion

Enabled in V2 config. These are zero-cost improvements (features already extracted from pipeline). No ablation done yet.

### Suggested Config Changes for V3

| Setting | Current | Proposed | Rationale |
|---------|---------|----------|-----------|
| `lr` | 0.001 | 0.0003-0.0005 | Lower initial LR reduces pos_weight overshoot |
| `lr_scheduler` | cosine | reduce_on_plateau | Keeps LR alive long enough for tail classes |
| `pos_weight` | per-threshold sqrt | Remove or use only per-item not per-threshold | Stops 0→1 distribution flip |
| `session_loss_weight` | 1.0 | 0.5-0.8 | May be over-regularizing on A01-excluded data |

---

## 4. Code Quality Notes

### What's solid

- **GroupedParticipantDataset**: Clean separation of session loading, participant grouping, and collation. `_make_dummy_session` handles missing sessions gracefully.
- **NaN gradient guard**: `train_one_epoch_grouped` checks `grads_finite` before `optimizer.step()` — rare but important. Functions correctly — V2 training had zero NaN events over 17 epochs.
- **Length-bucketed batching**: `build_length_bucketed_batches` reduces padding waste from ~72% to ~20%.
- **A2 decode strategy selection**: Auto-selects among argmax/monotonic/expectation with/without calibration on val set. Validation log shows this works correctly.
- **Config design**: YAML + CLI override is simple and effective. Flattening `feature_selection` into top-level config avoids nested dict access.
- **Session-level loss excludes A01**: `is_clinical = session_types[valid_session_mask] != 0` correctly excludes the neutral reading passage from auxiliary loss.

### What needs attention

1. **`use_coral` bug** at [runner.py:953](common/runner.py#L953): `use_coral` is an undefined variable — references `_defaults.use_coral` which doesn't exist in `FeatureConfig`. Should be `bool(cfg.get("use_coral", False))`. If `FeatureConfig` default is always False, CORAL is never actually used unless the config YAML explicitly has `use_coral: true`. **Check whether V2 actually ran with CORALHead or A2OrdinalHead.**

2. **Joint mode A1 submission bug** at [runner.py:1359](common/runner.py#L1359): Uses `"a1_preds" in dir()` which always returns `False` in this scope. The A1 predictions are never written to CSV in joint mode.

3. **pyarrow not installed**: eGeMAPS parquet loading fails silently. The JSON fallback works but only if the data has a `features` key.

4. **No ablation tracking**: Multiple LLM feature versions (V1 34-dim, V2 41-dim, V3 74-dim) exist without systematic comparison. Need a controlled ablation once ordinal collapse is fixed.

5. **Config drift risk**: The config YAML specifies `use_coral: true` and `gamma: 0.0` but the V2 training log shows no CORAL-specific validation and uses `a2_ordinal_loss` directly. The relationship between `use_coral` config, `CORALHead`, `A2OrdinalHead`, and the loss functions needs clarification.

---

## 5. Recommended Experiment Priority

**Updated 2026-05-24 based on V2 training results.**

| Priority | Experiment | Expected Impact | Effort |
|----------|------------|-----------------|--------|
| **P0** | Fix `use_coral` bug | **Critical** — may reveal V2 ran with wrong head | ~30m |
| **P0** | Remove pos_weight + oversample score=2/3 | **High** — stop 0→1 flip, give tail classes direct signal | ~2h |
| **P0** | Replace cosine LR with ReduceLROnPlateau | **High** — keep learning alive beyond epoch 7 | ~1h |
| **P0** | Investigate CLL=0.0 | **High** — may be dead code in MTL version | ~1h |
| **P1** | Per-threshold bias in CORALHead | Medium — let each threshold adapt independently | ~3h |
| **P1** | Fix joint mode A1 submission bug | Low | ~1h |
| **P2** | LLM bottleneck width + version ablation | Medium — once ordinal collapse is fixed | ~4h |
| **P2** | Body pose / global motion ablation | Low | ~2h |
| **P3** | Cross-modal attention reattempt | Medium — already attempted and reverted | ~4h |
