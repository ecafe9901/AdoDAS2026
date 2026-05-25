# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ADODAS2026 baseline — multimodal (audio + video) deep learning for adolescent mental health assessment. Two tracks:

- **A1**: Binary classification of Depression / Anxiety / Stress (3 outputs, sigmoid)
- **A2**: 21 ordinal item predictions with scores 0–3 (ordinal regression with 3 thresholds per item)

## Commands

```bash
# Environment: use base anaconda3 python (conda SSL is broken on this machine)
~/anaconda3/bin/python train.py --task a2 --config tasks/a2/default.yaml

# Inference
~/anaconda3/bin/python infer.py --task a2 --checkpoint <output_dir>/runs/<run_name>/checkpoints/best.pt --split test_hidden

# Override config values via CLI flags
~/anaconda3/bin/python train.py --task a2 --config tasks/a2/default.yaml --epochs 10 --lr 5e-4 --batch_size 16
```

There are no tests or linting setup in this repository. pyarrow is not installed (SSL blocks pip/conda installs) — eGeMAPS parquet loading will fail, but JSON fallback tries `pooled.json` (which lacks feature values on this dataset; remove `egemaps` from audio_features if needed).

## Architecture

```
train.py  →  common/runner.py:main()         # training entry point
infer.py                                     # standalone inference entry point
```

### Data flow

1. Features are pre-extracted per-session at `<feature_root>/<split>/<school>/<class>/<pid>/<modality>/<feature_set>/<session>/`
2. `GroupedParticipantDataset` groups the 4 sessions (A01, B01, B02, B03) per participant, loads `.npz` sequences and pooled features, aligns all modalities to a common 40ms time grid. A01 is a neutral reading passage (no clinical content); B01/B02/B03 are clinical interviews.
3. `grouped_collate_fn` flattens all sessions across a batch into one flat tensor — dummy zero sessions with zero masks fill in for missing sessions
4. `MTCNBackbone` processes each flat session: per-modality GroupAdapter → ModalityFusion (audio/video separately) → TCN (dilated residual conv1d blocks, 4 layers = 1.24s receptive field) → ASP (Attentive Statistics Pooling with VAD + QC signals) → final fusion MLP with session embedding
5. `GroupedModel` reshapes session reps into `(B, 4, D)`, applies ParticipantAggregator (mean/mlp/attention) over sessions, plus an auxiliary SessionTypeClassifier
6. Training loss = main task loss + `session_loss_weight *` session-level loss (excludes A01) + `session_type_loss_weight *` session type CE. For joint mode, `a1_loss_weight *` A1 loss is added.
7. After training, thresholds/biases are calibrated on the validation set and saved under `<run_dir>/calibration/`

### Key modules

| Module | Purpose |
|---|---|
| `common/data/dataset.py` | `FeatureConfig`, time-grid alignment, `MultimodalDataset` (unused — grouped path preferred) |
| `common/data/grouped_dataset.py` | `GroupedParticipantDataset` (4 sessions per participant), `grouped_collate_fn`, `build_length_bucketed_batches` |
| `common/data/feature_io.py` | Low-level `.npz` / parquet / JSON feature loading |
| `common/models/mtcn_backbone.py` | `MTCNBackbone`, `TCN`, `ASP`, `GroupAdapter`, `ModalityFusion` |
| `common/models/grouped_model.py` | `GroupedModel`, `ParticipantAggregator`, `SessionTypeClassifier`, `CORALHead` |
| `common/models/heads.py` | `A1Head`, `A2OrdinalHead`, loss functions (`a1_loss`, `a2_ordinal_loss` with Focal Loss `gamma` param) |
| `common/runner.py` | Training loop, validation, calibration (bias/threshold grid search), CSV submission generation |
| `common/utils/metrics.py` | `binary_f1`, `mean_qwk`, `mean_mae`, `macro_auroc`, `per_item_qwk` |
| `public_pipeline/` | Feature extraction pipeline (not needed for training/inference) |
| `scripts/` | LLM feature extraction (`extract_llm_v1.py`) and calibration (`calibrate_llm_features.py`) |

### Joint A1+A2 training

When `--task joint`, the model trains with both A2 (ordinal) and A1 (binary D/A/S) heads sharing the same backbone. The checkpoint stores both `head_state_dict` (A2) and `a1_head_state_dict` (A1). Inference in joint mode outputs two CSV files: `submission_joint_*.csv` and `submission_a1_*.csv`.

### LLM features

34-dim calibrated features from transcripts (V1 format): 21 DASS self-report item predictions + 13 behavioral markers (valence, emotion distribution, engagement, richness). Extracted by `scripts/extract_llm_v1.py`, calibrated by `scripts/calibrate_llm_features.py`. Stored as `.npy` files at `llm_feature_dir/<split>/<school>_<class>_<pid>.npy`.

For A2, the 21 DASS dimensions create a shortcut (predicting questionnaire items from questionnaire-derived features). The LLM features are projected through a 64-dim bottleneck before concatenation with the participant representation.

### A2 decoding

Three strategies: `argmax` (sigmoid > 0.5 count), `expectation` (sum of sigmoids rounded), `monotonic` (monotonically constrained class probabilities). When `decode_method: auto`, the best strategy is selected on the validation set. Post-hoc threshold calibration (per-item offset grid search) on 6 strategy combinations (raw/calibrated × 3 decode methods).

## Configuration

YAML config files at `tasks/<a1|a2>/default.yaml`. CLI args override YAML values. `feature_selection` block is flattened into the top-level config dict. `FeatureConfig` dataclass defines defaults.

Key config parameters (A2 V1 baseline):

| Param | Value | Notes |
|---|---|---|
| `d_model` / `d_shared` | 256 | Model width |
| `tcn_layers` | 4 | Receptive field = 1.24s |
| `batch_size` | 32 | |
| `session_loss_weight` | 1.0 | Only applied to B01/B02/B03 (A01 excluded in code) |
| `session_type_loss_weight` | 0.15 | |
| `gamma` | 2.0 | Focal loss gamma for A2 ordinal BCE |
| `weight_decay` | 0.02 | Reduced from 0.05 to fix score=3 blindness |
| `label_smoothing` | 0.0 | Reduced from 0.1 (was masking rare class signal) |
| `use_coral` | true | CORAL head with learned thresholds |
| `use_llm_features` | true | 34-dim transcript features |
| `mask_policy` | and_core | Both core features must be valid per frame |
| `core_audio` | mel_mfcc, ssl_embed | |
| `core_video` | vision_ssl_embed, qc_stats | |
| `dropout` | 0.3 | |

## Data characteristics

- 4,200 training participants, all complete with 4 sessions (16,800 session rows)
- ~600 validation participants
- A2 label distribution: 70.3% score=0, 22.6% score=1, 4.7% score=2, 2.4% score=3 — severe imbalance
- Sessions: A01 (reading, neutral), B01/B02/B03 (clinical interviews)
- B03 systematically shorter (avoidance signal)

## Known issues

- **`use_coral` bug** at `common/runner.py:953`: `use_coral` is an undefined variable — should be `bool(cfg.get("use_coral", False))`
- **pyarrow not installed**: eGeMAPS parquet loading fails. Remove `egemaps` from `audio_features` if running without pyarrow.
- **conda SSL broken**: Use `~/anaconda3/bin/python` directly; packages are pre-installed in base conda env.
- **No NaN gradient guard**: single NaN gradient can silently poison training. The previous training loop had this check but it was reverted along with Phase 1+2+4+5 changes.

## Reverted experimental changes

The following were tested in commit `d2ed754` ("Phase 1+2+4+5"), found problematic, and reverted:

- **Cross-modal attention**: Single-head MHA between audio/video before ASP, disabled for T > 1200 frames (effectively dead code for most sessions)
- **SpecAugment**: Time/freq masking on mel_mfcc only
- **d_model 320**: OOM on 16GB GPU, scaled back to 256
- **NaN gradient safety check**: Should be re-added independently
