# Dataset Analysis — ADODAS 2026

## Summary

| Property | Train | Val | Test |
|---|---|---|---|
| Participants | 4,200 | 600 | 1,200 |
| Sessions/participant | 4 (A01/B01/B02/B03) | 4 | 4 |
| Session completeness | 100% | 100% | — |
| Label consistency across sessions | 100% | 100% | — |

---

## 1. Label Distribution

### A2 — 21 ordinal items (0–3)

| Score | Train | Val |
|---|---|---|
| 0 | 70.3% | 69.4% |
| 1 | 22.6% | 23.4% |
| 2 | 4.7% | 4.7% |
| 3 | 2.4% | 2.5% |

Median sum score: 4.0 (out of 63). 85% of participants have sum < 20.

### A1 — Binary D/A/S

| Class | Train | Val |
|---|---|---|
| Depression | 21.5% | 21.5% |
| Anxiety | 28.2% | 28.3% |
| Stress | 12.9% | 13.0% |

---

## 2. Extreme Class Imbalance — Critical

- 70.3% of A2 scores are 0. Only 2.4% are score=3.
- The model can achieve low BCE without ever predicting score=2 or 3.
- pos_weight overcorrects → oscillation (A2: 0↔1 swings, A1: all-zero plateaus).
- Without pos_weight, A1 never escapes all-zero predictions (BCE optimum).
- A2's ordinal structure provides natural incentive to predict non-zero; A1's binary task does not.

---

## 3. School Distribution Shift — Critical

### Proportion mismatch

| Comparison | χ² | p-value |
|---|---|---|
| Train vs Val | 21.2 | **0.012** |
| Train vs Test | 31.7 | **0.0002** |
| Val vs Test | 17.9 | **0.037** |

All three pairwise comparisons are statistically significant. The data was not stratified by school when splitting into train/val/test.

### Per-school score variation — up to 6×

| School | Train N | Train mean score | % zeros |
|---|---|---|---|
| SCH_003 | 433 | 0.110 | 91.8% |
| SCH_008 | 456 | 0.171 | 86.1% |
| SCH_005 | 680 | 0.675 | 48.5% |

SCH_003 and SCH_005 differ by 6× in mean score. Schools may differ in demographics, interview protocols, or labeling practices.

### Proportion shifts in test set

| School | Train | Test | Shift |
|---|---|---|---|
| SCH_005 | 16.2% | 19.1% | **+2.9%** (highest-score school) |
| SCH_009 | 13.6% | 8.8% | **−4.8%** (moderate-score school) |

Test set has more high-score participants than training proportion would predict. Validation QWK may not generalize to test.

---

## 4. Item Difficulty Variation — up to 4×

| Item | % zeros | Entropy | Difficulty |
|---|---|---|---|
| d21 | 84.8% | 0.54 | Hardest — almost never triggers |
| d04 | 78.5% | 0.65 | |
| d09 | 54.4% | 1.08 | Easiest — most varied |

Easy items have 3-4× more signal than hard items. A shared CORAL head treats all items equally.

---

## 5. A01 Label Contamination

A01 (neutral reading passage) has the same clinical labels (y_D, y_A, y_S, d01–d21) as B01/B02/B03. The model is forced to predict depression scores from reading aloud — an impossible task. This injects noise into the session auxiliary loss.

---

## 6. Session Representation Collapse

Session variance (A1): 0.001→0.010 — model produces nearly identical representations for all 4 sessions. `session_loss_weight=0.2` in A1 is too low to encourage session differentiation. In A2, `session_loss_weight=0.5` produces healthier session variance.

---

## 7. Per-Item Correlation

Items are moderately correlated (r=0.25–0.73). d17↔d21 strongest (r=0.73). d14↔d21 weakest (r=0.25). The latent DASS structure exists but is not redundant — items carry complementary signals.

---

## 8. School as Wasted Training Data

SCH_003 contributes 433 participants (10.3% of training data) with 91.8% zeros. These participants provide almost no learning signal but consume ~35 GB of I/O per epoch. SCH_006 and SCH_008 are similarly sparse (80%+ zeros).
