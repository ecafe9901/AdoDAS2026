# Improvement Proposals — ADODAS 2026

Based on dataset analysis and training experiments (V1, V2, P0-P3).

---

## Phase 1: Monitoring & Observability (do now, ~30 min)

### 1.1 Per-School Validation QWK

**File**: `common/runner.py` — `validate_grouped()`

Add per-school QWK breakdown to detect if the model is overfitting to school-specific recording conditions.

```python
school_qwks = {}
for sch in set(batch_schools):
    mask = (schools_np == sch)
    if mask.sum() > 5:
        school_qwks[sch] = mean_qwk(preds_np[mask], labels_np[mask])
log.info(f"  Per-school QWK: " + " ".join(f"{s}={v:.3f}" for s, v in sorted(school_qwks.items())))
```

**Cost**: ~15 lines. **Benefit**: Answers "does QWK vary by school?"

### 1.2 Per-Item QWK Trend Tracking

**File**: `common/utils/run_metadata.py`

Save per-item QWK per epoch for later analysis. Identifies items that stay at random chance (likely the high-zero items like d21, d04).

```python
meta.set_extra(f"epoch_{epoch}_item_qwk", per_item_qwk(preds, labels).tolist())
```

**Cost**: ~5 lines. **Benefit**: Identifies items that never improve.

---

## Phase 2: Data Flow Fixes (do next, ~2 hours)

### 2.1 Exclude A01 from Session Auxiliary Loss

**File**: `common/runner.py` — `train_one_epoch_grouped()`

A01 (reading passage) has no clinical content but shares the same labels as B01-B03. Excluding it from session auxiliary loss prevents noisy gradients.

```python
# session_types: 0=A01, 1=B01, 2=B02, 3=B03
is_clinical = session_types[valid_session_mask] != 0
if is_clinical.any():
    sess_loss = a2_ordinal_loss(s_logits[is_clinical], s_targets[is_clinical], ...)
else:
    sess_loss = torch.tensor(0.0)
```

**Cost**: ~8 lines. **Benefit**: Removes impossible prediction targets from loss. Validated in V2 — needs re-application after rollback.

### 2.2 School-Aware Embedding

**Files**: `common/models/grouped_model.py`, `common/data/grouped_dataset.py`, `common/runner.py`

Add a learnable school embedding (10 schools × 16-dim) to the participant representation. Lets the model learn site-specific baselines while keeping the clinical backbone site-agnostic.

```python
# grouped_model.py
self.school_emb = nn.Embedding(10, 16)

# forward():
school_bias = self.school_emb(school_idx)
participant_repr = torch.cat([participant_repr, school_bias], dim=-1)
```

```python
# grouped_dataset.py — add school_idx to participant dict
SCHOOL_TO_IDX = {f"SCH_{i:03d}": i for i in range(1, 11)}
info["school_idx"] = SCHOOL_TO_IDX.get(str(info["anon_school"]), 0)
```

**Cost**: ~30 lines. **Benefit**: Explicitly models school effects rather than learning them implicitly.

### 2.3 School-Aware pos_weight

**File**: `common/runner.py` — `compute_a2_pos_weight()`

SCH_003 (92% zeros) needs different pos_weight than SCH_005 (48% zeros). Compute per-school pos_weight to avoid systematic under/over-correction.

```python
def compute_a2_pos_weight_per_school(manifest_path):
    df = pd.read_csv(manifest_path)
    pw_per_school = {}
    for sch, group in df.groupby("anon_school"):
        pw_per_school[sch] = compute_pos_weight(group)
    return pw_per_school
```

**Cost**: ~20 lines. **Benefit**: Fairer treatment across schools.

---

## Phase 3: Training Stability (validated, ongoing)

### 3.1 Gentle pos_weight for A2

**File**: `common/runner.py` — `compute_a2_pos_weight()`

**Status**: Implemented.

```python
max_clip = {0: 1.0, 1: 2.0, 2: 3.0}  # k=0 no push, k=1 gentler, k=2 gentler
```

Let the model learn the 0↔1 boundary freely. Only nudge score≥2 and score≥3.

### 3.2 Gentle pos_weight for A1

**File**: `common/runner.py` — `_compute_pos_weight_a1()`

**Status**: Implemented. Max clip reduced from 4.0 → 2.0. Primary effect: Stress weight 2.59→2.0.

### 3.3 NaN Gradient Guard

**File**: `common/runner.py` — `train_one_epoch_grouped()`

**Status**: Implemented. Checks gradient finiteness before `optimizer.step()`, skips poisoned batches.

### 3.4 Remove pos_weight for A2 (use_pos_weight=false)

**Finding**: A2 trains best without any pos_weight (QWK=0.204 vs 0.08 with pos_weight). CORAL learns thresholds naturally from data. pos_weight causes oscillation.

**Status**: Applied in config. Keep as default for A2.

---

## Phase 4: Efficiency (deferred)

### 4.1 SCH_003 Downsampling

SCH_003 has 433 participants with 91.8% zeros — almost no learning signal. Randomly downsample to reduce I/O.

```python
if school == "SCH_003" and random.random() < 0.5:
    continue  # skip 50% of this school's participants
```

**Cost**: ~10 lines. **Benefit**: ~5% I/O and computation savings per epoch.

### 4.2 Label Smoothing for A1

**File**: `tasks/a1/default.yaml`

```yaml
label_smoothing: 0.05  # was 0.0
```

Prevents the model from becoming overly confident on negative predictions. Helps escape the all-zero plateau.

---

## Implementation Roadmap

| # | Item | Status | Lines | Phase |
|---|---|---|---|---|
| 1.1 | Per-school validation QWK | Not started | ~15 | Monitoring |
| 1.2 | Per-item QWK tracking | Not started | ~5 | Monitoring |
| 2.1 | Exclude A01 from session loss | Not started (rolled back) | ~8 | Data fix |
| 2.2 | School-aware embedding | Not started | ~30 | Data fix |
| 2.3 | School-aware pos_weight | Not started | ~20 | Data fix |
| 3.1 | Gentle A2 pos_weight | ✅ Implemented | — | Stability |
| 3.2 | Gentle A1 pos_weight | ✅ Implemented | — | Stability |
| 3.3 | NaN gradient guard | ✅ Implemented | — | Stability |
| 3.4 | A2 use_pos_weight=false | ✅ Applied | — | Stability |
| 4.1 | SCH_003 downsampling | Deferred | ~10 | Efficiency |
| 4.2 | A1 label_smoothing | Deferred | 1 | Efficiency |
