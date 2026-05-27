# A1 Optimization Summary

## Background

In joint A1+A2 training, the A1 head (3 binary targets: Depression/Anxiety/Stress) was originally
dead — F1=0.000 in both adversarial and non-adversarial joint runs. This document summarizes
the optimization path that revived A1 and the weight-tuning experiments.

## Changes Made

### 1. Unclip pos_weight (Phase A — effective, kept)

**File:** `common/runner.py` — `compute_a2_pos_weight()`

The original `max_clip = {0: 1.0, 1: 2.0, 2: 3.0}` suppressed gradients for all ordinal thresholds.
Removing the clip (cap at 10.0) not only helped A2 rare-class prediction but also revived A1.
The A1 head shares the backbone with A2; when A2 receives proper gradient signal, the shared
representation becomes richer, benefiting A1.

**Result:** A1 F1 0.00 → 0.32 (MLP), 0.00 → 0.37 (transformer)

### 2. Transformer aggregator with residual connection (Phase B — effective, kept)

**File:** `common/models/grouped_model.py` — `TransformerAggregator`

Replaced mean/mlp session pooling with a 2-layer, 4-head TransformerEncoder with [CLS] token
and learned positional encoding. A residual mean-pool connection prevents warmup collapse.
The richer session-level representation benefits both A1 and A2.

**Result:** A1 F1 0.32 → 0.37 (transformer vs MLP in joint training)

### 3. a1_loss_weight tuning (tested, w=0.3 kept as optimal)

**File:** `common/runner.py`, `tasks/a2/default.yaml`

| Weight | A1 F1 (epoch 7) | A2 QWK (epoch 7) | Result |
|--------|-----------------|-------------------|--------|
| 0.3 (original) | 0.221 | 0.082 | Stable, best balance |
| 1.0 | 0.303 | 0.065 | A1 higher early, QWK catches up at epoch 12 (0.110 vs 0.121) |
| 2.0 | 0.000 | 0.000 | Collapse — A1 loss dominates, backbone degrades |

**Decision:** w=0.3 kept. Higher weights do not improve final QWK; the A2 ordinal signal
naturally provides enough representation for A1 to learn as an auxiliary task.

## Failed Approaches

### A1-only training (rejected)

Training with `--task a1` (no A2 joint) with transformer aggregator:
- F1 stuck at 0.34 (calibrated), raw F1=0.000 (all zeros)
- Only Anxiety showed intermittent signal; Depression/Stress always zero
- The A2 ordinal task provides critical representation learning that A1 cannot achieve alone

### Focal loss (rejected)

gamma=1.0 and gamma=2.0 both caused model collapse to all-zero predictions.
The focal term (1-pt)^gamma downweights easy negatives so aggressively that
the model settles into a degenerate solution.

## Current State

- **A1 F1:** 0.37 (joint + transformer, `a1_loss_weight=0.3`)
- **A2 QWK:** 0.226 (calibrated), score 2 at 0.5%, score 3 at 2.5% on test
- **Config:** `tasks/a2/default.yaml` with `aggregator: transformer`, `use_pos_weight: true`,
  `a1_loss_weight: 0.3`

## Key Takeaways

1. A1 benefits more from A2's rich ordinal signal than from being trained alone
2. The default `a1_loss_weight=0.3` is near-optimal; A2's 63 targets dominate gradient flow
   and this is actually beneficial for the shared representation
3. Unclipping pos_weight for A2 indirectly revived A1 by improving backbone gradients
4. Transformer with residual connection boosts both tasks
