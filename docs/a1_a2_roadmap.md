# A1 & A2 Improvement Roadmap

## Current State

| Metric | Value | Target | Gap |
|--------|:---:|:---:|:---:|
| A2 QWK (calibrated) | 0.225 | 0.27 | 0.045 |
| A2 score 2 (raw) | 0.5% | ≥3% | 2.5pp |
| A2 score 3 (raw) | 2.2% | ≥1.5% | met |
| A1 F1 | 0.424 | 0.531 | 0.107 |
| A1 Depression | weak | — | — |
| A1 Stress | weak | — | — |

**Key findings:**
- Two-stage training works: Stage 2 freezes backbone+A2, trains only A1 head (33K params)
  → A1 F1 0.413→0.424 in 36 min
- Loss weighting alone cannot fix A2 rare-class prediction (gamma=1.0/2.0 cause collapse)
- Transformer aggregator with residual connection enables stable training and score 2 emergence
- A1 ceiling is limited by backbone representation quality, not A1 head capacity
- School annotation bias (SCH_003 D=0% vs SCH_005 D=41%) is a primary A1 bottleneck

## Bottlenecks (Prioritized)

### P0: A2 Rare-Class Ordinal Collapse

Score 2 and 3 are almost never predicted in raw logits (only via calibration threshold shift).
The 3 independent BCE thresholds lack ordinal constraint: raising logit for threshold 3
disturbs already-learned thresholds 1-2. A consistent ordinal structure would let rare
classes receive gradient without destabilization.

### P1: A1 School Annotation Bias

SCH_003 has 433 participants with D=3.9%, SCH_008 has 456 with D=6.1%, while SCH_005
has 680 with D=41.3%. These 10× differences are annotation inconsistency, not real
epidemiology. A1 head receives "SCH_003 features → no depression" as a shortcut, learning
school identity instead of clinical features.

### P2: Backbone Representation for A1

Stage 2 freezes backbone → A1 plateaus at 0.424. Improving the backbone's ordinal
representation (P0) should raise this ceiling. Additional features (LLM behavioral
markers, body pose) may also help.

## Improvement Phases

### Phase C: Ordinal Loss Fix (A2-focused)

| Step | Description | Expected |
|------|-------------|:---:|
| C1 | Replace independent BCE with constrained ordinal loss (CORN or ordinal CE) | Score 2≥3%, score 3≥1.5% in raw |
| C2 | Add per-threshold bias in CORALHead or A2OrdinalHead | Tail classes get independent offset |
| C3 | Full 40-epoch Stage 1 training | A2 QWK raw ≥0.21 |
| C4 | Stage 2: freeze backbone → train A1 head | A1 ≥0.45 |

**Risk:** Medium. New loss function may need tuning. Mitigation: 5-epoch smoke test first.

### Phase D: A1 School Bias Mitigation

| Step | Description | Expected |
|------|-------------|:---:|
| D1 | Mask SCH_003/008 A1 loss (keep A2) | Remove largest annotation bias sources |
| D2 | Per-school A1 calibration (threshold per school) | Each school gets independent D/A/S threshold |
| D3 | A1 per-school bias in head | Small learned bias per school, regularized |

**Risk:** Low. Only affects A1 head, backbone unchanged.

### Phase E: Feature & Calibration Polish

| Step | Description | Expected |
|------|-------------|:---:|
| E1 | Enable LLM behavioral features (13-dim) | +0.01 A2 QWK, +0.02 A1 F1 |
| E2 | Per-school A2 calibration (inference-time) | +0.005-0.01 A2 QWK |
| E3 | Hyperparameter sweep (LR, weight_decay, dropout) | Minor gains |

**Risk:** Low. All proven in prior experiments.

## Execution Order

```
Phase C (ordinal loss) ── Stage 1: train backbone+A2
  │                          Stage 2: freeze → train A1
  │
  ├── Phase D (A1 school bias) ── Stack on Phase C
  │     Stage 2 with SCH_003/008 A1 loss masked
  │
  └── Phase E (LLM + calibration) ── Final polish
```

## Expected Progression

| Milestone | A2 QWK | A1 F1 | Key Change |
|-----------|:---:|:---:|------|
| Current | 0.225 | 0.424 | Two-stage, transformer, pos_weight |
| Phase C | 0.25-0.27 | 0.45-0.48 | Ordinal loss fix |
| Phase D | 0.25-0.27 | 0.48-0.50 | School bias masking |
| Phase E | 0.27-0.29 | 0.50-0.53 | LLM features + calibration |
