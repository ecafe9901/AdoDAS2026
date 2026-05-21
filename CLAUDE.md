# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ADODAS2026 baseline — multimodal (audio + video) deep learning for adolescent mental health assessment. Two tracks:

- **A1**: Binary classification of Depression / Anxiety / Stress (3 outputs, sigmoid)
- **A2**: 21 ordinal item predictions with scores 0–3 (ordinal regression with 3 thresholds per item)

## Commands

```bash
# Create environment
conda env create -f envs/adodas.yaml && conda activate adodas

# Train
python train.py --task a1 --config tasks/a1/default.yaml
python train.py --task a2 --config tasks/a2/default.yaml

# Override config values via CLI flags (most hypers are exposed)
python train.py --task a1 --config tasks/a1/default.yaml --epochs 10 --lr 5e-4 --batch_size 32

# Inference
python infer.py --task a1 --checkpoint <output_dir>/runs/<run_name>/checkpoints/best.pt
python infer.py --task a2 --checkpoint <output_dir>/runs/<run_name>/checkpoints/best.pt --split test_hidden
```

There are no tests or linting setup in this repository.

## Architecture

```
train.py  →  common/runner.py:main()         # training entry point
infer.py                                     # standalone inference entry point
```

### Data flow

1. Features are pre-extracted per-session and stored in `<feature_root>/<split>/<school>/<class>/<pid>/<modality>/<feature_set>/<session>/`
2. `GroupedParticipantDataset` groups the 4 sessions (A01, B01, B02, B03) per participant, loads `.npz` sequences and pooled features, aligns all modalities to a common 40ms time grid
3. `grouped_collate_fn` flattens all sessions across a batch into one flat tensor — dummy zero sessions with zero masks fill in for missing sessions
4. `MTCNBackbone` processes each flat session: per-modality GroupAdapter → ModalityFusion (audio/video separately) → TCN (dilated residual conv1d blocks) → ASP (Attentive Statistics Pooling with VAD + QC signals) → final fusion MLP with session embedding
5. `GroupedModel` reshapes session reps into `(B, 4, D)`, applies ParticipantAggregator (mean/mlp/attention) over sessions, plus an auxiliary SessionTypeClassifier on individual session reps
6. Training loss = main task loss + `session_loss_weight *` session-level loss + `session_type_loss_weight *` session type cross-entropy
7. After training, thresholds/biases are calibrated on the validation set and saved under `<run_dir>/calibration/`

### Key modules

| Module | Purpose |
|---|---|
| `common/data/dataset.py` | `FeatureConfig`, `MultimodalDataset`, time-grid alignment, per-sample loading |
| `common/data/grouped_dataset.py` | `GroupedParticipantDataset` (4 sessions per participant), `grouped_collate_fn` |
| `common/data/feature_io.py` | Low-level `.npz` / parquet / JSON feature loading |
| `common/models/mtcn_backbone.py` | `MTCNBackbone`, `TCN`, `ASP`, `GroupAdapter`, `ModalityFusion` |
| `common/models/grouped_model.py` | `GroupedModel`, `ParticipantAggregator`, `SessionTypeClassifier`, `CORALHead` |
| `common/models/heads.py` | `A1Head`, `A2OrdinalHead`, loss functions (`a1_loss`, `a2_ordinal_loss`) |
| `common/runner.py` | Training loop, validation, calibration (bias/threshold grid search), CSV submission generation |
| `common/utils/metrics.py` | `binary_f1`, `mean_qwk`, `mean_mae`, `macro_auroc`, `per_item_qwk` |
| `public_pipeline/` | Feature extraction pipeline (not needed for training/inference — features are pre-extracted) |

### Future improvement ideas (not yet implemented)

**Transcript metadata (meta.json, segments.json):**
- `audio_duration`: per-session audio length (B03 systematically shorter → avoidance signal)
- `has_exact_segments` / `n_segments`: ASR quality gate
- `question_id` (B01/B02/B03): embedding for session-type-aware processing
- Cross-session duration ratios (B03/B01, B02-B03 length) → emotional engagement proxy
- SenseVoice emotion tags (NEUTRAL 87%, SAD 4.5%, HAPPY 3%, EMO_UNKNOWN 5%): time-aligned audio emotion → add as per-frame bias in ASP attention

**Architecture improvements:**
- Cross-modal attention (audio↔video) before ASP — biggest single gain potential
- Transformer-based session aggregator (4 session self-attention)
- d_model 256→384: headroom for larger capacity (VRAM only 9GB/16GB)

### Configuration

YAML config files are at `tasks/<a1|a2>/default.yaml`. CLI args override YAML values. The `feature_selection` block in YAML is flattened into the top-level config dict. `FeatureConfig` dataclass defines defaults for all feature-related settings.

### A2 decoding

Three decoding strategies for converting ordinal logits to integer scores (0–3): `argmax`, `expectation`, `monotonic`. When `decode_method: auto`, the best strategy is selected on the validation set. Post-hoc threshold calibration (per-item offset grid search) can further improve QWK.
