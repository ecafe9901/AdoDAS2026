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

### 1.2 Per-Class Validation QWK

**File**: `common/runner.py`, `common/data/grouped_dataset.py`

Extend school monitoring to class level. Add `anon_classes` to batch output (collate function already has the data), then log per-class QWK.

```python
# grouped_dataset.py — grouped_collate_fn
"anon_classes": [b["anon_class"] for b in batch],

# runner.py — validate_grouped
all_classes.extend(batch.get("anon_classes", []))
# After validation loop — same as school pattern
```

**Cost**: ~10 lines. **Benefit**: Detects if specific classes (e.g., SCH_005/CLS_0107 with 1.014 mean) are driving validation metrics.

### 1.3 Per-Item QWK Trend Tracking

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

**Cost**: ~30 lines. **Benefit**: Explicitly models school effects rather than learning them implicitly. Class effects are nested within schools — the school embedding captures most class-level variance. Adding per-class embeddings (249 classes) would overfit — not recommended.

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

## Phase 3: Feature Selection (validated findings)

### 3.0 body_pose + global_motion — Disable for A2

**Finding**: These 2 video features (72 extra dims) add ~2 GB VRAM and dilute A2 signal. A2 QWK with them enabled plateaued at 0.04 vs 0.20 without. Keep disabled for A2. For A1, not yet tested in isolation.

**Status**: Verified harmful. Should remain disabled in A2 config.

---

## Phase 4: Training Stability (validated, ongoing)

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

### 4.1 Emotional State Stratification

**File**: `common/data/grouped_dataset.py` — batch sampler

Use the Emotional state change column (State=2 has 4× higher DASS scores) for stratified batch sampling. Ensures each batch has proportional representation from State=1/2/3 (40%/19%/41%).

```python
# Stratify by emotional state during batch building
state_groups = {1: [], 2: [], 3: []}
for p in participants:
    state_groups[p["emotional_state"]].append(p)
# Sample proportionally from each group per batch
```

**Cost**: ~15 lines. **Benefit**: Prevents model from overfitting to low-score participants (State=1/3, 80% of data) and never seeing high-score State=2 cases. Particularly valuable for A1 where Stress=12.9% positive rate.

### 4.2 SCH_003 Downsampling

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
| 1.1 | Per-school validation QWK | ✅ Implemented | ~15 | Monitoring |
| 1.2 | Per-class validation QWK | Not started | ~10 | Monitoring |
| 1.3 | Per-item QWK tracking | ✅ Implemented | ~5 | Monitoring |
| 2.1 | Exclude A01 from session loss | Not started (rolled back) | ~8 | Data fix |
| 2.2 | School-aware embedding | Not started | ~30 | Data fix |
| 2.3 | School-aware pos_weight | Not started | ~20 | Data fix |
| 3.0 | body_pose+global_motion disabled A2 | ✅ Verified harmful | — | Feature |
| 3.1 | Gentle A2 pos_weight | ✅ Implemented | — | Stability |
| 3.2 | Gentle A1 pos_weight (max clip 2.0) | ✅ Implemented | — | Stability |
| 3.3 | NaN gradient guard | ✅ Implemented | — | Stability |
| 3.4 | A2 use_pos_weight=false | ✅ Applied | — | Stability |
| 4.1 | Emotional state stratification | Not started | ~15 | Data fix |
| 4.2 | SCH_003 downsampling | Deferred | ~10 | Efficiency |
| 4.3 | A1 label_smoothing | Deferred | 1 | Efficiency |
