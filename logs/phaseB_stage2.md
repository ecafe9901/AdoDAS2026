2026-05-27 19:50:02 [INFO] Logging to /mnt/data/datasets/AdoDAS/output/a2/runs/joint__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg__v-ssl-vit-mae-base__mask-andcore__pw__seed42__20260527_195002/logs/train_grouped_joint_20260527_195002.log
2026-05-27 19:50:02 [INFO] Device: cuda
2026-05-27 19:50:02 [INFO] Task: joint
2026-05-27 19:50:02 [INFO] Run name: joint__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg__v-ssl-vit-mae-base__mask-andcore__pw__seed42__20260527_195002
2026-05-27 19:50:02 [INFO] Mask policy: and_core
2026-05-27 19:50:04 [INFO] batch_size=24, num_workers=4
2026-05-27 19:50:04 [INFO] Building length-bucketed batches (reduces padding waste 72% -> ~20%) ...
2026-05-27 19:50:06 [INFO] Joint A1+A2 training: A2 + A1
2026-05-27 19:50:06 [INFO] Model params: 7,003,292
2026-05-27 19:50:06 [INFO] AMP enabled (BF16)
2026-05-27 19:50:06 [INFO] A2 pos_weight shape: torch.Size([1, 21, 3]), A1 pw [D/A/S]: 1.91/1.59/2.60
2026-05-27 19:50:06 [INFO] AMP disabled for Stage 2
2026-05-27 19:50:06 [INFO] Stage 2: loading backbone from /mnt/data/datasets/AdoDAS/output/a2/runs/joint__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg__v-ssl-vit-mae-base__mask-andcore__pw__seed42__20260527_170034/checkpoints/best.pt
2026-05-27 19:50:07 [INFO] Stage 2: backbone + A2 frozen, A1 head re-initialized
2026-05-27 19:50:08 [INFO] Scheduler: warmup=0 -> cosine, total=20
2026-05-27 19:50:08 [INFO] Grad clip: 1.0
2026-05-27 19:50:08 [INFO] School weights: SCH_001=1.34 SCH_002=1.05 SCH_003=0.45 SCH_004=0.87 SCH_005=1.39 SCH_006=0.84 SCH_007=1.04 SCH_008=0.64 SCH_009=1.25 SCH_010=1.12
2026-05-27 19:50:08 [INFO] Session loss weight: 0.5
2026-05-27 19:50:08 [INFO] Session type loss weight: 0.15
2026-05-27 19:50:08 [INFO] EarlyStopping: patience=10, metric=primary, mode=max
2026-05-27 19:50:08 [INFO] Label smoothing: 0.05
2026-05-27 19:50:08 [INFO] Feature noise std: 0.01
2026-05-27 19:50:08 [INFO] Session drop prob: 0.05
2026-05-27 19:50:08 [INFO] ==========================================================================================
2026-05-27 19:50:08 [INFO] ==========================================================================================
  output = torch._nested_tensor_from_mask(
2026-05-27 19:53:33 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 19:53:33 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 19:53:33 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 19:53:33 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 19:53:33 [INFO]     A1 F1: 0.4128
2026-05-27 19:53:33 [INFO]     1/ 20 | 9.94e-04 |   0.9337   |  0.6637  |  0.1989  |  0.4081  | A1 0.413 | 3m 24s ETA 1h 04m 52s VRAM 2.3G *
2026-05-27 19:53:33 [INFO]   >>> New best QWK=0.1989 saved at epoch 1.
2026-05-27 19:56:56 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 19:56:56 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 19:56:56 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 19:56:56 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 19:56:56 [INFO]     A1 F1: 0.4083
2026-05-27 19:56:56 [INFO]     2/ 20 | 9.76e-04 |   0.9230   |  0.6637  |  0.1989  |  0.4081  | A1 0.408 | 3m 22s ETA 1h 01m 08s VRAM 2.3G
2026-05-27 20:00:22 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:00:22 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:00:22 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:00:22 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:00:22 [INFO]     A1 F1: 0.4242
2026-05-27 20:00:23 [INFO]     3/ 20 | 9.46e-04 |   0.9215   |  0.6637  |  0.1989  |  0.4081  | A1 0.424 | 3m 26s ETA 58m 00s VRAM 2.3G
2026-05-27 20:03:40 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:03:40 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:03:40 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:03:40 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:03:40 [INFO]     A1 F1: 0.4105
2026-05-27 20:03:40 [INFO]     4/ 20 | 9.05e-04 |   0.9177   |  0.6637  |  0.1989  |  0.4081  | A1 0.410 | 3m 17s ETA 54m 05s VRAM 2.3G
2026-05-27 20:06:56 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:06:57 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:06:57 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:06:57 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:06:57 [INFO]     A1 F1: 0.4121
2026-05-27 20:06:57 [INFO]     5/ 20 | 8.54e-04 |   0.9139   |  0.6637  |  0.1989  |  0.4081  | A1 0.412 | 3m 16s ETA 50m 24s VRAM 2.3G
2026-05-27 20:10:19 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:10:19 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:10:19 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:10:19 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:10:19 [INFO]     A1 F1: 0.4120
2026-05-27 20:10:19 [INFO]     6/ 20 | 7.94e-04 |   0.9119   |  0.6637  |  0.1989  |  0.4081  | A1 0.412 | 3m 22s ETA 47m 04s VRAM 2.3G
2026-05-27 20:13:33 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:13:33 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:13:33 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:13:33 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:13:33 [INFO]     A1 F1: 0.4120
2026-05-27 20:13:33 [INFO]     7/ 20 | 7.27e-04 |   0.9128   |  0.6637  |  0.1989  |  0.4081  | A1 0.412 | 3m 14s ETA 43m 28s VRAM 2.3G
2026-05-27 20:16:46 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:16:46 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:16:46 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:16:46 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:16:46 [INFO]     A1 F1: 0.4105
2026-05-27 20:16:47 [INFO]     8/ 20 | 6.55e-04 |   0.9114   |  0.6637  |  0.1989  |  0.4081  | A1 0.410 | 3m 13s ETA 39m 57s VRAM 2.3G
2026-05-27 20:19:59 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:19:59 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:19:59 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:19:59 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:19:59 [INFO]     A1 F1: 0.4041
2026-05-27 20:19:59 [INFO]     9/ 20 | 5.79e-04 |   0.9084   |  0.6637  |  0.1989  |  0.4081  | A1 0.404 | 3m 12s ETA 36m 29s VRAM 2.3G
2026-05-27 20:23:13 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:23:13 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:23:13 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:23:13 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:23:13 [INFO]     A1 F1: 0.4239
2026-05-27 20:23:13 [INFO]    10/ 20 | 5.01e-04 |   0.9134   |  0.6637  |  0.1989  |  0.4081  | A1 0.424 | 3m 13s ETA 33m 05s VRAM 2.3G
2026-05-27 20:26:26 [INFO]     pred dist: 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:26:26 [INFO]     GT   dist: 0=69.4% 1=23.4% 2=4.7% 3=2.5%
2026-05-27 20:26:26 [INFO]     top3: d16=0.281 d08=0.275 d15=0.240  |  bot3: d04=0.141 d09=0.129 d21=0.123
2026-05-27 20:26:26 [INFO]     per-school QWK: SCH_001=0.064 SCH_002=-0.076 SCH_003=0.166 SCH_004=-0.022 SCH_005=0.175 SCH_006=0.336 SCH_007=0.022 SCH_008=0.016 SCH_009=-0.018 SCH_010=-0.002
2026-05-27 20:26:26 [INFO]     A1 F1: 0.4167
2026-05-27 20:26:26 [INFO]    11/ 20 | 4.22e-04 |   0.9095   |  0.6637  |  0.1989  |  0.4081  | A1 0.417 | 3m 12s ETA 29m 41s VRAM 2.3G
2026-05-27 20:26:26 [INFO]   EarlyStopping triggered at epoch 11 (patience=10, metric=primary)
2026-05-27 20:26:26 [INFO] ==========================================================================================
2026-05-27 20:26:26 [INFO] Loading best checkpoint for submission generation ...
2026-05-27 20:26:27 [INFO] Submission level: participant
2026-05-27 20:26:27 [INFO] Decode method: auto
2026-05-27 20:26:27 [INFO] Calibrating and selecting A2 decode strategy on val ...
2026-05-27 20:26:55 [INFO]   A2 decode comparison on val:
2026-05-27 20:26:55 [INFO]     argmax                 QWK=0.1326 MAE=0.3876 | 0=83.2% 1=16.8% 2=0.0% 3=0.0%
2026-05-27 20:26:55 [INFO]     monotonic              QWK=0.0696 MAE=0.3900 | 0=90.7% 1=9.3% 2=0.0% 3=0.0%
2026-05-27 20:26:55 [INFO]     expectation            QWK=0.1989 MAE=0.4081 | 0=69.2% 1=30.8% 2=0.0% 3=0.0%
2026-05-27 20:26:55 [INFO]     calibrated_argmax      QWK=0.2240 MAE=0.4792 | 0=53.6% 1=41.8% 2=4.6% 3=0.0%
2026-05-27 20:26:55 [INFO]     calibrated_monotonic   QWK=0.2253 MAE=0.4755 | 0=58.5% 1=38.0% 2=0.8% 3=2.7%
2026-05-27 20:26:55 [INFO]     calibrated_expectation QWK=0.2200 MAE=0.4167 | 0=63.8% 1=36.2% 2=0.0% 3=0.0%
2026-05-27 20:26:55 [INFO]   Selected A2 strategy: calibrated_monotonic (decode=monotonic, QWK=0.2253, MAE=0.4755)
2026-05-27 20:26:55 [INFO] Skipping submission generation after training; use infer.py for release inference.
2026-05-27 20:26:55 [INFO] Run complete: joint__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg__v-ssl-vit-mae-base__mask-andcore__pw__seed42__20260527_195002
2026-05-27 20:26:55 [INFO] Output dir: /mnt/data/datasets/AdoDAS/output/a2/runs/joint__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg__v-ssl-vit-mae-base__mask-andcore__pw__seed42__20260527_195002
