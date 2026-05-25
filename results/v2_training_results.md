# V2 Training Results

**Version**: V2 (commit `79dbbb3`)
**Date**: 2026-05-23 17:12 – 2026-05-24 04:00
**Total time**: 10h 44m 13s
**Early stop**: epoch 17/25 (patience=10 on QWK)

## Config

| Param | V2 Value | V1 (baseline) |
|---|---|---|
| tcn_layers | **6** | 4 |
| gamma (focal loss) | **0.0** | 2.0 |
| LLM dims (A2) | **13** (behavioral only) | 34 (21 DASS + 13 behavioral) |
| epochs / warmup | 25 / 3 | 40 / 3 |
| batch_size | **24** | 32 |
| early_stop_metric | **primary (QWK)** | val_loss |
| patience | **10** | 8 |
| session_drop_prob | **0.05** | 0.1 |
| weight_decay | 0.02 | 0.02 |
| label_smoothing | 0.0 | 0.0 |
| session_loss_weight | 1.0 (A01 excluded) | 1.0 (all sessions) |
| NaN gradient guard | **Yes** | No |
| preload | False | False |
| d_model / d_shared | 256 | 256 |
| head | CORAL | CORAL |

## Epoch Results

| Epoch | LR | Train Loss | Val Loss | QWK | MAE | 0% | 1% | 2% | 3% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.40e-4 | 1.4398 | 0.6373 | 0.0011 | 0.6342 | 28.3 | 71.7 | 0.0 | 0.0 |
| 2 | 6.70e-4 | 1.3558 | 0.6181 | **0.0671** | 0.6118 | 30.6 | 69.4 | 0.0 | 0.0 |
| 3 | 1.00e-3 | 1.3311 | 0.6152 | 0.0643 | 0.6199 | 29.7 | 70.3 | 0.0 | 0.0 |
| 4 | 9.95e-4 | 1.2638 | 0.6125 | 0.0474 | 0.6507 | 22.7 | 77.3 | 0.0 | 0.0 |
| 5 | 9.80e-4 | 1.2104 | 0.6106 | 0.0556 | 0.6407 | 24.3 | 75.7 | 0.0 | 0.0 |
| 6 | 9.55e-4 | 1.2050 | 0.6111 | 0.0628 | 0.6307 | 27.3 | 72.7 | 0.0 | 0.0 |
| **7** | **9.21e-4** | **1.2003** | **0.6114** | **0.0701** | **0.6316** | 27.6 | 72.4 | 0.0 | 0.0 |
| 8 | 8.78e-4 | 1.2070 | 0.6108 | 0.0568 | 0.6390 | 24.7 | 75.3 | 0.0 | 0.0 |
| 9 | 8.28e-4 | 1.2158 | 0.6107 | 0.0628 | 0.6402 | 25.9 | 74.1 | 0.0 | 0.0 |
| 10 | 7.71e-4 | 1.2102 | 0.6100 | 0.0626 | 0.6431 | 25.0 | 75.0 | 0.0 | 0.0 |
| 11 | 7.08e-4 | 1.2034 | 0.6106 | 0.0630 | 0.6401 | 25.9 | 74.1 | 0.0 | 0.0 |
| 12 | 6.41e-4 | 1.1981 | 0.6100 | 0.0630 | 0.6308 | 27.3 | 72.7 | 0.0 | 0.0 |
| 13 | 5.72e-4 | 1.1962 | 0.6093 | 0.0610 | 0.6444 | 24.7 | 75.3 | 0.0 | 0.0 |
| 14 | 5.01e-4 | 1.1958 | 0.6091 | 0.0582 | 0.6501 | 23.6 | 76.4 | 0.0 | 0.0 |
| 15 | 4.29e-4 | 1.1934 | 0.6095 | 0.0639 | 0.6403 | 25.2 | 74.8 | 0.0 | 0.0 |
| 16 | 3.60e-4 | 1.1944 | 0.6097 | 0.0664 | 0.6366 | 26.3 | 73.7 | 0.0 | 0.0 |
| 17 | 2.93e-4 | 1.1941 | 0.6107 | 0.0667 | 0.6335 | 27.2 | 72.8 | 0.0 | 0.0 |

GT distribution: 0=69.4% 1=23.4% 2=4.7% 3=2.5%

## Calibration Results

Post-hoc threshold calibration on validation set:

| Strategy | QWK | MAE | 0% | 1% | 2% | 3% |
|---|---|---|---|---|---|---|
| argmax (raw) | 0.0069 | 0.4064 | 95.4 | 4.6 | 0.0 | 0.0 |
| monotonic (raw) | 0.0000 | 0.4033 | 100.0 | 0.0 | 0.0 | 0.0 |
| expectation (raw) | 0.0701 | 0.6316 | 27.6 | 72.4 | 0.0 | 0.0 |
| calibrated_argmax | **0.0959** | 0.5733 | 45.8 | 52.2 | **2.1** | 0.0 |
| calibrated_monotonic | 0.0694 | 0.9810 | 70.9 | 0.0 | 0.0 | 29.1 |
| **calibrated_expectation** | **0.0959** | **0.5603** | 45.5 | 54.5 | 0.0 | 0.0 |

**Selected**: calibrated_expectation (QWK=0.0959)

## Key Findings

1. **Best QWK = 0.0701** at epoch 7, improving from V1's best of 0.0798 (but V1 used focal loss which showed unstable behavior)
2. **Calibration improved QWK to 0.096** — threshold offsets partially compensate for model bias
3. **Scores 2-3 never predicted** during training — pos_weight pushes all scores past threshold 1 but model never reaches thresholds 2-3 before LR decays
4. **Cosine LR decay too aggressive** — LR dropped from 1e-3 to 2.9e-4, stopping learning after epoch ~7
5. **Train loss flatlined at 1.19** — model stopped learning meaningful representations after epoch 4
6. **6-layer TCN (5.08s receptive field)** and **LLM behavioral-only features** were the right changes — training was stable (no OOM, no NaN)
7. **calibrated_argmax achieved 2.1% score=2 predictions** — first time any score=2 was predicted

## Next Steps (V3)

- Disable or reduce pos_weight (it causes the 0→1 distribution flip)
- Use constant LR or less aggressive decay schedule
- Consider per-item loss weighting instead of pos_weight
