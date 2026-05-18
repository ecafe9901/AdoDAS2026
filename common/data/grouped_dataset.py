from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import (
    SESSIONS, SESSION_TO_IDX, ITEM_COLS, A1_COLS,
    FeatureConfig, align_to_grid,
)
from .feature_io import SequenceData, load_egemaps_pooled, load_sequence

log = logging.getLogger(__name__)


class GroupedParticipantDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
        session_drop_prob: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.feature_root)
        self.session_drop_prob = float(session_drop_prob)

        manifest = pd.read_csv(manifest_path)

        group_cols = ["anon_school", "anon_class", "anon_pid"]
        grouped = manifest.groupby(group_cols)

        self.participants: list[dict[str, Any]] = []
        for (school, cls, pid), group in grouped:
            sess_rows = {}
            for _, row in group.iterrows():
                sess = str(row["session"])
                sess_rows[sess] = row

            any_row = group.iloc[0]
            y_a1 = np.array([float(any_row.get(c, -1)) for c in A1_COLS], dtype=np.float32)
            y_a2 = np.array([float(any_row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32)

            self.participants.append({
                "anon_school": str(school),
                "anon_class": str(cls),
                "anon_pid": str(pid),
                "sess_rows": sess_rows,
                "y_a1": y_a1,
                "y_a2": y_a2,
            })

        self._feature_dims: dict[str, int] | None = None
        self._cache: list[dict | None] | None = None
        self._text_cache: dict[str, list[int]] = {}  # cache_key -> char_ids
        self._char_vocab: dict[str, int] | None = None

    @property
    def feature_dims(self) -> dict[str, int]:
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        info = self.participants[0]
        sess_rows = info["sess_rows"]
        any_sess = list(sess_rows.keys())[0]
        row = sess_rows[any_sess]
        dims: dict[str, int] = {}
        for name, seq in self._load_raw_groups(row, "audio").items():
            dims[name] = seq.features.shape[1]
        for name, seq in self._load_raw_groups(row, "video").items():
            dims[name] = seq.features.shape[1]
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, self.split,
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                dims["egemaps"] = len(eg)
        return dims

    def _load_raw_groups(self, row, modality: str) -> dict[str, SequenceData]:
        cfg = self.cfg
        feat_list = cfg.audio_sequence_features if modality == "audio" else cfg.video_features
        groups: dict[str, SequenceData] = {}
        for feat_name in feat_list:
            tag: str | None = None
            if feat_name == "ssl_embed":
                tag = cfg.audio_ssl_model_tag
            elif feat_name == "vision_ssl_embed":
                tag = cfg.video_ssl_model_tag
            try:
                seq = load_sequence(
                    self.root, self.split,
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]),
                    modality, feat_name, str(row["session"]),
                    model_tag=tag,
                )
                groups[feat_name] = seq
            except FileNotFoundError:
                pass
        return groups

    def _compute_modality_mask(
        self, mask_parts, mask_names, core_names, policy, T
    ) -> np.ndarray:
        if not mask_parts:
            return np.zeros(T, dtype=bool)
        if policy == "or":
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "and_core":
            core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
            if core_masks:
                return np.all(np.stack(core_masks), axis=0)
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "require_k":
            k = max(1, len(core_names))
            stacked = np.stack(mask_parts)
            return np.sum(stacked, axis=0) >= k
        raise ValueError(f"Unknown mask_policy: {policy!r}")

    def _load_single_session(self, row) -> dict[str, Any] | None:
        """Load features for a single session. Returns None on failure."""
        cfg = self.cfg
        try:
            audio_raw = self._load_raw_groups(row, "audio")
            video_raw = self._load_raw_groups(row, "video")

            all_groups = {}
            for k, v in audio_raw.items():
                all_groups[f"audio/{k}"] = v
            for k, v in video_raw.items():
                all_groups[f"video/{k}"] = v

            if not all_groups:
                return None

            aligned_feats, aligned_masks, grid_ms, T = align_to_grid(
                all_groups, cfg.grid_step_ms, cfg.tolerance_ms
            )

            audio_groups: dict[str, torch.Tensor] = {}
            video_groups: dict[str, torch.Tensor] = {}
            audio_mask_parts, audio_mask_names = [], []
            video_mask_parts, video_mask_names = [], []

            for key, feat in aligned_feats.items():
                modality, name = key.split("/", 1)
                mask = aligned_masks[key]
                t = torch.from_numpy(feat.astype(np.float32))
                if modality == "audio":
                    audio_groups[name] = t
                    audio_mask_parts.append(mask)
                    audio_mask_names.append(name)
                else:
                    video_groups[name] = t
                    video_mask_parts.append(mask)
                    video_mask_names.append(name)

            mask_audio = self._compute_modality_mask(
                audio_mask_parts, audio_mask_names, cfg.core_audio, cfg.mask_policy, T
            )
            mask_video = self._compute_modality_mask(
                video_mask_parts, video_mask_names, cfg.core_video, cfg.mask_policy, T
            )

            vad_signal = np.zeros(T, dtype=np.float32)
            if "audio/vad" in aligned_feats:
                v = aligned_feats["audio/vad"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["audio/vad"].astype(np.float32)
            elif "video/vad_agg" in aligned_feats:
                v = aligned_feats["video/vad_agg"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["video/vad_agg"].astype(np.float32)

            qc_quality = np.zeros(T, dtype=np.float32)
            if "video/qc_stats" in aligned_feats:
                v = aligned_feats["video/qc_stats"]
                qc_quality = v[:, 0].astype(np.float32) * aligned_masks["video/qc_stats"].astype(np.float32)

            dims = self.feature_dims
            audio_pooled_groups: dict[str, torch.Tensor] = {}
            pooled_presence: dict[str, bool] = {}
            if "egemaps" in cfg.audio_pooled_features:
                egemaps = load_egemaps_pooled(
                    self.root, self.split,
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]), str(row["session"]),
                )
                audio_pooled_groups["egemaps"] = (
                    torch.from_numpy(egemaps) if egemaps is not None
                    else torch.zeros(dims.get("egemaps", 88))
                )
                pooled_presence["egemaps"] = egemaps is not None

            for name in cfg.audio_features:
                if name not in audio_groups and name not in cfg.audio_pooled_features and name in dims:
                    audio_groups[name] = torch.zeros(T, dims[name])
            for name in cfg.video_features:
                if name not in video_groups and name in dims:
                    video_groups[name] = torch.zeros(T, dims[name])

            # Text char IDs (transcript)
            if cfg.use_text_features:
                char_ids = self._load_text_embedding(
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]), str(row["session"]),
                )

            session_idx = SESSION_TO_IDX.get(str(row["session"]), 0)

            result = {
                "audio_groups": audio_groups,
                "audio_pooled_groups": audio_pooled_groups,
                "video_groups": video_groups,
                "mask_audio": torch.from_numpy(mask_audio),
                "mask_video": torch.from_numpy(mask_video),
                "vad_signal": torch.from_numpy(vad_signal),
                "qc_quality": torch.from_numpy(qc_quality),
                "audio_pooled_present": pooled_presence,
                "session_idx": session_idx,
                "seq_len": T,
                "session": str(row["session"]),
            }
            if cfg.use_text_features:
                result["text_char_ids"] = char_ids
            return result
        except Exception as e:
            log.debug(f"Failed to load session {row.get('session', '?')} for {row.get('anon_pid', '?')}: {e}")
            return None

    def _load_text_embedding(self, school: str, cls: str, pid: str,
                              session: str) -> list[int]:
        """Load cached char IDs (must be pre-encoded by pre_encode_texts)."""
        cache_key = "__A01__" if session == "A01" else f"{pid}_{session}"
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        return [0]  # PAD only

    def pre_encode_texts(self) -> int:
        """Build char vocab and pre-compute char IDs for all transcripts."""
        if not self.cfg.use_text_features:
            return 0
        from .text_features import load_transcript, _build_char_vocab, _char_ids

        all_texts = []
        keys = []
        for info in self.participants:
            pid = info["anon_pid"]
            school = info["anon_school"]
            cls = info["anon_class"]
            for sess in ["A01", "B01", "B02", "B03"]:
                cache_key = f"{pid}_{sess}"
                if cache_key in self._text_cache:
                    continue
                text = load_transcript(self.root, self.split, school, cls, pid, sess)
                if text:
                    all_texts.append(text)
                    keys.append(cache_key)
                else:
                    self._text_cache[cache_key] = [0]  # mark empty with PAD id

        if not all_texts:
            return 0

        self._char_vocab = _build_char_vocab(all_texts, min_freq=2)
        for text, key in zip(all_texts, keys):
            self._text_cache[key] = _char_ids(text, self._char_vocab)

        # Also cache under __A01__ for shared A01 text
        for key in list(self._text_cache.keys()):
            if key.endswith("_A01"):
                self._text_cache["__A01__"] = self._text_cache[key]
                break

        log.info(f"Char vocab: {len(self._char_vocab)}, encoded {len(all_texts)} transcripts")
        return len(all_texts)

    def __len__(self) -> int:
        return len(self.participants)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and self._cache[idx] is not None:
            sample = self._cache[idx]
        else:
            sample = self._load_participant(idx)

        if self.split == "train" and self.session_drop_prob > 0.0:
            return self._apply_session_dropout(sample)
        return sample

    def _load_participant(self, idx: int) -> dict[str, Any]:
        info = self.participants[idx]
        sessions_data = []
        session_valid = []

        for sess_name in SESSIONS:
            if sess_name in info["sess_rows"]:
                data = self._load_single_session(info["sess_rows"][sess_name])
                if data is not None:
                    sessions_data.append(data)
                    session_valid.append(True)
                else:
                    sessions_data.append(None)
                    session_valid.append(False)
            else:
                sessions_data.append(None)
                session_valid.append(False)

        # LLM features (per-participant, loaded from .npy)
        llm_features = None
        if self.cfg.use_llm_features and self.cfg.llm_feature_dir:
            llm_dir = Path(self.cfg.llm_feature_dir) / self.split
            llm_path = llm_dir / f"{info['anon_school']}_{info['anon_class']}_{info['anon_pid']}.npy"
            if llm_path.exists():
                try:
                    llm_features = torch.from_numpy(np.load(llm_path))
                except Exception:
                    pass

        result = {
            "sessions": sessions_data,
            "session_valid": np.array(session_valid, dtype=bool),
            "y_a1": torch.from_numpy(info["y_a1"]),
            "y_a2": torch.from_numpy(info["y_a2"]),
            "anon_pid": info["anon_pid"],
            "anon_school": info["anon_school"],
            "anon_class": info["anon_class"],
            "session_names": SESSIONS,
        }
        if llm_features is not None:
            result["llm_features"] = llm_features
        return result

    def _apply_session_dropout(self, sample: dict[str, Any]) -> dict[str, Any]:
        valid_indices = [
            idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
            if is_valid and sample["sessions"][idx] is not None
        ]
        if len(valid_indices) <= 1 or np.random.random() >= self.session_drop_prob:
            return sample

        drop_idx = int(np.random.choice(valid_indices))
        sessions = list(sample["sessions"])
        sessions[drop_idx] = None
        session_valid = np.array(sample["session_valid"], copy=True)
        session_valid[drop_idx] = False

        return {
            **sample,
            "sessions": sessions,
            "session_valid": session_valid,
        }

    def preload(self, desc: str | None = None) -> float:
        n = len(self)
        if desc is None:
            desc = f"Preload {self.split}"
        self._cache = [None] * n
        errors = 0
        for i in tqdm(range(n), desc=desc, dynamic_ncols=True):
            try:
                self._cache[i] = self._load_participant(i)
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    log.warning(f"Preload: participant {i} failed: {exc}")
        if errors > 0:
            log.warning(f"Preload: {errors}/{n} participants failed")
        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {n - errors}/{n} participants ({gb:.1f} GB in RAM)")
        return gb

    def _estimate_cache_bytes(self) -> int:
        total = 0
        if self._cache is None:
            return 0
        for sample in self._cache:
            if sample is None:
                continue
            for sess in sample.get("sessions", []):
                if sess is None:
                    continue
                for v in sess.values():
                    if isinstance(v, torch.Tensor):
                        total += v.nelement() * v.element_size()
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, torch.Tensor):
                                total += vv.nelement() * vv.element_size()
        return total

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None


def build_length_bucketed_batches(
    dataset: GroupedParticipantDataset,
    batch_size: int,
    num_buckets: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """Build batches where participants have similar max session lengths.

    Dynamically adjusts batch size per bucket: short-session buckets use
    larger batches, long-session buckets use smaller batches. Targets
    the same memory footprint per batch.
    """
    import random as _random
    import math
    rng = _random.Random(seed)

    n = len(dataset)
    max_lens = []
    for idx in range(n):
        max_t = _fast_max_session_len(dataset, idx)
        max_lens.append((idx, max_t, 4.0))  # (idx, max_T, weight=4.0)

    max_lens.sort(key=lambda x: x[1])

    bucket_size = max(1, n // num_buckets)
    batches = []
    for b in range(0, n, bucket_size):
        bucket = max_lens[b:b + bucket_size]
        rng.shuffle(bucket)

        # Dynamic batch size: aim for ~batch_size * median_T total frames per batch
        bucket_Ts = [t for _, t, _ in bucket]
        median_T = bucket_Ts[len(bucket_Ts) // 2]
        target_total = batch_size * 400  # 400 ≈ overall median T
        dyn_bs = max(2, min(batch_size, int(target_total / max(median_T, 1))))
        dyn_bs = min(dyn_bs, len(bucket))  # can't exceed bucket size

        indices = [idx for idx, _, _ in bucket]
        for start in range(0, len(indices), dyn_bs):
            batch = indices[start:start + dyn_bs]
            if len(batch) >= 2:
                batches.append(batch)

    rng.shuffle(batches)
    return batches


def _fast_max_session_len(dataset: GroupedParticipantDataset, idx: int) -> int:
    """Get max session length by reading only mel_mfcc/A01 timestamps.

    A01 (reading passage) is always the longest session, so its T is the max.
    """
    import numpy as np
    info = dataset.participants[idx]
    root = str(dataset.root)
    split = dataset.split

    if "A01" not in info["sess_rows"]:
        return 100  # fallback

    row = info["sess_rows"]["A01"]
    seq_path = (
        f"{root}/{split}/{row['anon_school']}/{row['anon_class']}/{row['anon_pid']}"
        f"/audio/mel_mfcc/A01/sequence.npz"
    )
    try:
        with np.load(seq_path) as data:
            if "timestamps_ms" in data:
                return len(data["timestamps_ms"])
            for key in data.keys():
                arr = data[key]
                if hasattr(arr, 'shape') and len(arr.shape) >= 1:
                    return arr.shape[0]
    except Exception:
        pass
    return 100  # fallback


def grouped_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    all_sessions = []  
    session_types = [] 
    session_valid_list = []
    flat_pids = []
    flat_sess_names = []

    for b_idx, sample in enumerate(batch):
        session_valid_list.append(sample["session_valid"])
        for s_idx, sess_data in enumerate(sample["sessions"]):
            if sess_data is not None:
                all_sessions.append(sess_data)
                session_types.append(s_idx)
                flat_pids.append(sample["anon_pid"])
                flat_sess_names.append(SESSIONS[s_idx])
            else:
                dims = batch[0]["sessions"]
                ref = None
                for s in sample["sessions"]:
                    if s is not None:
                        ref = s
                        break
                if ref is None:
                    for other in batch:
                        for s in other["sessions"]:
                            if s is not None:
                                ref = s
                                break
                        if ref is not None:
                            break

                if ref is not None:
                    dummy = _make_dummy_session(ref)
                    all_sessions.append(dummy)
                    session_types.append(s_idx)
                    flat_pids.append(sample["anon_pid"])
                    flat_sess_names.append(SESSIONS[s_idx])

    if not all_sessions:
        raise RuntimeError("No valid sessions in batch")

    n_flat = len(all_sessions)
    T_max = max(s["seq_len"] for s in all_sessions)

    audio_names = list(all_sessions[0]["audio_groups"].keys())
    pooled_audio_names = list(all_sessions[0]["audio_pooled_groups"].keys())
    video_names = list(all_sessions[0]["video_groups"].keys())

    def _pad_groups(names, key):
        result = {}
        for n in names:
            D = all_sessions[0][key][n].shape[-1]
            t = torch.zeros(n_flat, T_max, D)
            for i, s in enumerate(all_sessions):
                L = s["seq_len"]
                t[i, :L] = s[key][n]
            result[n] = t
        return result

    def _pad_1d(key, dtype=torch.float32):
        t = torch.zeros(n_flat, T_max, dtype=dtype)
        for i, s in enumerate(all_sessions):
            L = s["seq_len"]
            t[i, :L] = s[key]
        return t

    pad_mask = torch.ones(n_flat, T_max, dtype=torch.bool)
    for i, s in enumerate(all_sessions):
        pad_mask[i, :s["seq_len"]] = False

    # Text char IDs (list of lists, encoded by CharTextEncoder in backbone)
    text_char_ids = None
    has_text = any("text_char_ids" in s for s in all_sessions)
    if has_text:
        text_char_ids = [
            s.get("text_char_ids", [0]) for s in all_sessions
        ]

    flat_batch = {
        "audio_groups": _pad_groups(audio_names, "audio_groups"),
        "audio_pooled_groups": {
            name: torch.stack([s["audio_pooled_groups"][name] for s in all_sessions])
            for name in pooled_audio_names
        },
        "video_groups": _pad_groups(video_names, "video_groups"),
        "mask_audio": _pad_1d("mask_audio", torch.bool),
        "mask_video": _pad_1d("mask_video", torch.bool),
        "pad_mask": pad_mask,
        "vad_signal": _pad_1d("vad_signal"),
        "qc_quality": _pad_1d("qc_quality"),
        "audio_pooled_present": {
            name: torch.tensor(
                [s["audio_pooled_present"].get(name, False) for s in all_sessions],
                dtype=torch.bool,
            )
            for name in pooled_audio_names
        },
        "session_idx": torch.tensor([s["session_idx"] for s in all_sessions], dtype=torch.long),
        "seq_len": torch.tensor([s["seq_len"] for s in all_sessions], dtype=torch.long),
        "anon_pid": flat_pids,
        "session": flat_sess_names,
        "text_char_ids": text_char_ids,
    }

    # LLM features (per-participant pooled features)
    llm_features_list = [b.get("llm_features") for b in batch]
    llm_features = None
    if any(f is not None for f in llm_features_list):
        # Fill missing with zeros
        dim = next(f.shape[0] for f in llm_features_list if f is not None)
        llm_features = torch.stack([
            f if f is not None else torch.zeros(dim)
            for f in llm_features_list
        ])

    return {
        "flat_batch": flat_batch,
        "participant_y_a1": torch.stack([b["y_a1"] for b in batch]),
        "participant_y_a2": torch.stack([b["y_a2"] for b in batch]),
        "session_valid": torch.from_numpy(np.stack(session_valid_list)),
        "session_types": torch.tensor(session_types, dtype=torch.long),
        "n_participants": B,
        "anon_pids": [b["anon_pid"] for b in batch],
        "anon_schools": [b["anon_school"] for b in batch],
        "anon_classes": [b["anon_class"] for b in batch],
        "flat_sessions": flat_sess_names,
        "flat_pids": flat_pids,
        "llm_features": llm_features,
    }


def _make_dummy_session(ref: dict[str, Any]) -> dict[str, Any]:
    """Create a zero-filled dummy session matching reference dims."""
    T = 1  # minimal length
    audio_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["audio_groups"].items()}
    video_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["video_groups"].items()}
    dummy = {
        "audio_groups": audio_groups,
        "audio_pooled_groups": {
            k: torch.zeros_like(v) for k, v in ref["audio_pooled_groups"].items()
        },
        "video_groups": video_groups,
        "mask_audio": torch.zeros(T, dtype=torch.bool),
        "mask_video": torch.zeros(T, dtype=torch.bool),
        "vad_signal": torch.zeros(T),
        "qc_quality": torch.zeros(T),
        "audio_pooled_present": {
            k: False for k in ref["audio_pooled_groups"].keys()
        },
        "session_idx": 0,
        "seq_len": T,
        "session": "A01",
    }
    if "text_char_ids" in ref:
        dummy["text_char_ids"] = [0]  # PAD id
    return dummy
