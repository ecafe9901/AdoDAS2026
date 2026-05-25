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

## 4. Class-Level Distribution Shift — Worse than Schools

### Class Size Variation

The 249 classes (mean 17 students, range 1–41) have extreme internal variation — classes of 1–3 students have zero statistical reliability.

### Within-School Class Variation — up to 4×

| School | Classes | Min mean | Max mean | Range | CV |
|---|---|---|---|---|---|
| SCH_003 | 31 | 0.000 | 0.362 | 0.362 | 0.77 |
| SCH_005 | 34 | 0.345 | 1.429 | 1.083 | 0.34 |
| SCH_008 | 29 | 0.000 | 0.388 | 0.388 | 0.74 |
| SCH_010 | 41 | 0.000 | 1.000 | 1.000 | 0.44 |

SCH_005 has a 4× difference in mean score between its lowest and highest classes (0.345 → 1.429). Some classes within the same school are systematically different — potentially different recording setups, labeling practices, or demographics.

### Extreme Class Outliers

| School/Class | Students | Mean Score | %≥2 | Issue |
|---|---|---|---|---|
| SCH_003/CLS_0140 | 20 | 0.000 | 0% | All zeros — wasted data |
| SCH_005/CLS_0107 | 24 | 1.014 | 23% | 50% above school mean |
| SCH_005/CLS_0233 | 6 | 1.357 | 40% | 2× school mean |
| SCH_005/CLS_0148 | 28 | 0.345 | — | Half of school mean |
| SCH_001/CLS_0024 | 12 | 1.099 | 31% | 2× school mean |

### Class Overlap Across Splits

203/249 classes shared across all 3 splits. 6 train-only, 0 val-only, 1 test-only. Good overlap — class-level distribution shift is less about missing classes and more about internal score variation.

### Implication

The "school effect" is partially driven by class-level clustering. If a batch contains 8/12 participants from one high-scoring class (SCH_005/CLS_0107), the model may learn class-specific recording conditions rather than clinical signals. School embeddings capture most of this variance, but extreme class outliers can bias per-batch gradients.

---

## 5. Item Difficulty Variation — up to 4×

| Item | % zeros | Entropy | Difficulty |
|---|---|---|---|
| d21 | 84.8% | 0.54 | Hardest — almost never triggers |
| d04 | 78.5% | 0.65 | |
| d09 | 54.4% | 1.08 | Easiest — most varied |

Easy items have 3-4× more signal than hard items. A shared CORAL head treats all items equally.

---

## 6. A01 Label Contamination

A01 (neutral reading passage) has the same clinical labels (y_D, y_A, y_S, d01–d21) as B01/B02/B03. The model is forced to predict depression scores from reading aloud — an impossible task. This injects noise into the session auxiliary loss.

---

## 7. Session Representation Collapse

Session variance (A1): 0.001→0.010 — model produces nearly identical representations for all 4 sessions. `session_loss_weight=0.2` in A1 is too low to encourage session differentiation. In A2, `session_loss_weight=0.5` produces healthier session variance.

---

## 8. Per-Item Correlation

Items are moderately correlated (r=0.25–0.73). d17↔d21 strongest (r=0.73). d14↔d21 weakest (r=0.25). The latent DASS structure exists but is not redundant — items carry complementary signals.

---

## 10. Emotional State Change — Strong Non-Linear Signal (Hidden from Test)

The manifest CSV files contain 5 extra questionnaire columns beyond the DASS labels:

| Column | Values | Coverage |
|---|---|---|
| Family structure | 1-6 | 98.6% |
| Only child status | 0/1 | 100% |
| Parental favoritism | 1-3 | 65.1% |
| Academic performance change | 1-3 | 100% |
| Emotional state change | 1-3 | 100% |

None of these columns exist in the test set manifest.

### Emotional State Change — U-Shaped, 4× Score Difference

| State | Participants | A2 Sum Mean | %≥1 | %≥2 | Depression | Anxiety | Stress |
|---|---|---|---|---|---|---|---|
| 1 | 1,684 (40%) | 4.79 | 18.3% | 3.3% | 9.9% | 16.3% | 5.2% |
| **2** | **809 (19%)** | **19.19** | **63.5%** | **20.6%** | **58.7%** | **65.6%** | **40.7%** |
| 3 | 1,707 (41%) | 6.41 | 24.9% | 4.4% | 15.5% | 22.3% | 7.3% |

**ANOVA F=765.25, p=0.0000**. State=2 participants have 4× higher DASS scores than State=1. All 21 items are individually significant (p<0.0001). The relationship is U-shaped (State=1 low, State=2 high, State=3 low), which explains why simple Pearson r=0.068 missed it — correlation assumes linearity.

### Random Forest: R²=0.286 (Emotional state = 83% of feature importance)

A non-linear model using all 5 columns achieves R²=0.286 for predicting A2 sum score. Emotional state change alone dominates at 83% importance.

### Practical Use — Stratified Sampling

Since these columns are absent from test data, they cannot be used as features. But Emotional state change can guide **stratified batch sampling**: ensure each training batch has proportional representation from State=1/2/3. Prevents the model from overfitting to State=1/3 (80% of participants) while never encountering high-score State=2 cases.

---

## 11. body_pose + global_motion — Harmful for A2

These 2 video features (72 extra dimensions) were tested after re-enabling them from the upstream config:

| Run | QWK at Epoch 6-8 | Trend |
|---|---|---|
| Without body_pose/global_motion | 0.063→0.204 (peak) | Climbing |
| With body_pose/global_motion | 0.044 (plateaued) | Stuck |

The extra 72 feature dimensions increase VRAM usage (~2 GB) and appear to dilute the signal-to-noise ratio. For A2, these features should remain disabled. For A1, they may still have value (not yet tested in isolation).

---

## 9. School and Class as Wasted Training Data (continued)

SCH_003 contributes 433 participants (10.3% of training data) with 91.8% zeros. Within SCH_003, CLS_0140 has 20 students — all with mean score 0.000. These participants provide zero learning signal but consume ~35 GB of I/O per epoch.

SCH_005 has the opposite problem — some classes (CLS_0107, CLS_0233) score 2-4× above the school mean. The model receives inconsistent signals from the same school: half the classes suggest "high scoring school," half suggest "moderate."

Total wasted I/O: SCH_003 (10.3%) + sparse classes in SCH_006/SCH_008 (80%+ zeros) ≈ 20% of training data contributes near-zero gradient signal.
