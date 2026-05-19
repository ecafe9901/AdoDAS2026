#!/usr/bin/env python3
"""Pre-compute LLM-based features from transcripts using calibrated DeepSeek API.

V2 improvements:
  - Few-shot calibrated examples to prevent over-scoring
  - Quality score gating for ASR noise
  - 47-dim features: 21 DASS + 1 quality + psychological markers

Usage:
  python scripts/extract_llm_features.py \
    --manifest /path/to/manifest.csv \
    --feature_root /path/to/features \
    --split train \
    --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests
import urllib3

urllib3.disable_warnings()

log = logging.getLogger("extract_llm")

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = "deepseek-chat"

DASS_ITEMS = [f"d{i:02d}" for i in range(1, 22)]

_FEWSHOT_SYSTEM = """You are a clinical psychologist specializing in Chinese adolescent mental health.
Assess DASS-21 items from three speech transcripts (B01, B02, B03).

CALIBRATED EXAMPLES:

Example 1 (healthy, engaged):
B01:"我昨天过得很好，和同学一起去打篮球了"
B02:"上周最开心的是考试考了第一名"
B03:"没有特别悲伤的事"
→ All 0. Quality=1.0, valence=positive.

Example 2 (mild depressive symptoms):
B01:"昨天一般般，没什么特别的"
B02:"好像没有什么特别开心的"
B03:"跟朋友吵架了，有点难过"
→ d13=1 d17=1, others 0. Quality=0.9, valence=negative.

Example 3 (moderate academic stress, social withdrawal):
B01:"昨天很累，不想说话"
B02:"没有开心的事"
B03:"考试没考好，老师批评我了"
→ d01=2 d02=2 d05=2 d13=1 d14=1, others 0. Quality=0.8, valence=negative.

Example 4 (short answers, possible withdrawal - score CONSERVATIVELY):
B01:"嗯"
B02:"不知道"
B03:"没有"
→ All 0 or 1 only. Quality=0.3, valence=neutral.
When transcripts are minimal single-word answers, do NOT assume severe symptoms.

Example 5 (garbled/unintelligible text):
B01:"呵Q"
B02:"身份啊"
B03:"嗯"
→ All 0. Quality=0.0, valence=neutral.
Garbled text is NOT evidence of mental health issues.

SCORING RULES:
- Normal adolescent complaints (exams, arguments, tiredness) → 0-1, NOT 2-3
- Short transcripts (under 5 chars) → score 0-1 conservatively at most
- Garbled text → all zeros, quality_score=0.0
- Only score 2-3 with CLEAR explicit evidence (e.g., "I can't sleep at all", "I cry every day")
- quality_score: 0.0=garbled, 0.3=minimal/uninterpretable, 0.7=sparse but readable, 1.0=clear

Return ONLY a JSON object. No explanation text outside the JSON."""


def build_user_prompt(b01: str, b02: str, b03: str) -> str:
    return f"""B01 (How was your day?): "{b01}"
B02 (Happiest memory): "{b02}"
B03 (Saddest memory): "{b03}"

Return JSON with keys: d01-d21 (int 0-3), quality_score (float 0-1), valence (str positive/negative/neutral)."""


def call_deepseek(system: str, user: str, max_retries: int = 3) -> dict | None:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": user}], "max_tokens": 500, "temperature": 0.0}

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=60, verify=False)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = "\n".join(content.split("\n")[1:-1])
                return json.loads(content)
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                log.warning(f"API error {r.status_code}: {r.text[:200]}")
                time.sleep(1)
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def encode_features(result: dict, b01_len: int, b02_len: int, b03_len: int) -> np.ndarray:
    """Encode LLM result + transcript quality into 47-dim feature vector."""
    # 21 DASS scores
    dass = np.array([float(result.get(k, 0)) for k in DASS_ITEMS], dtype=np.float32)
    # Clamp to 0-3
    dass = np.clip(dass, 0, 3)

    # Quality score
    quality = float(result.get("quality_score", 0.5))

    # Transcript lengths (proxy for ASR quality)
    lengths = np.array([min(b01_len, 50), min(b02_len, 50), min(b03_len, 50)],
                       dtype=np.float32) / 50.0

    # Valence
    valence_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    valence = valence_map.get(result.get("valence", "neutral"), 0.0)

    # Combined: gating signal (low quality → downweight LLM features)
    gating = quality if quality > 0.3 else 0.1
    # If all 3 transcripts are very short (<5 chars), gate hard
    if max(b01_len, b02_len, b03_len) < 5:
        gating = 0.05

    # Psychological markers (8 dims)
    markers = np.array([
        valence,                          # -1/0/+1
        float(dass[12]),                  # d13 (depression item)
        float(dass[16]),                  # d17 (agitation item)
        float(dass[0]),                   # d01 (anxiety item)
        float(np.mean(dass[:7])),         # avg anxiety cluster
        float(np.mean(dass[7:14])),       # avg depression cluster
        float(np.mean(dass[14:])),        # avg stress cluster
        float(dass.sum()),                # total severity
    ], dtype=np.float32)

    # Normalize
    markers[4:7] = markers[4:7] / 3.0  # avg clusters to 0-1
    markers[7] = min(markers[7] / 30.0, 1.0)  # total severity normalized

    # Contrastive features (8 dims): B02 vs B03 comparisons
    contrastive = np.array([
        float(b02_len) / max(float(b03_len), 1),  # length ratio
        float(b01_len) / max((b02_len + b03_len) / 2, 1),  # B01 vs average
        1.0 if b02_len > 10 and b03_len < 5 else 0.0,  # happy-detailed, sad-minimal
        1.0 if b02_len < 5 and b03_len > 10 else 0.0,  # happy-absent, sad-detailed
        1.0 if max(b01_len, b02_len, b03_len) < 5 else 0.0,  # all minimal
        float(min(b01_len, b02_len, b03_len)) / max(float(max(b01_len, b02_len, b03_len)), 1),
        valence,
        gating,
    ], dtype=np.float32)

    return np.concatenate([dass, [quality], lengths, markers, contrastive])
    # 21 + 1 + 3 + 8 + 8 = 41 dims


def load_transcript(feature_root: Path, split: str, school: str, cls: str,
                    pid: str, session: str) -> str:
    path = feature_root / split / school / cls / pid / session / "clean_transcript.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


def process_manifest(manifest_path: Path, feature_root: Path, split: str,
                     output_dir: Path, limit: int = 0, delay: float = 0.5) -> int:
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(manifest_path)
    grouped = df.groupby(["anon_school", "anon_class", "anon_pid"])
    participants = [(str(s), str(c), str(p)) for (s, c, p), _ in grouped]

    if limit > 0:
        participants = participants[:limit]

    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for school, cls, pid in tqdm(participants, desc=f"Extract {split}"):
        out_path = out_dir / f"{school}_{cls}_{pid}.npy"
        if out_path.exists():
            success += 1
            continue

        b01 = load_transcript(feature_root, split, school, cls, pid, "B01")
        b02 = load_transcript(feature_root, split, school, cls, pid, "B02")
        b03 = load_transcript(feature_root, split, school, cls, pid, "B03")

        result = call_deepseek(_FEWSHOT_SYSTEM,
                               build_user_prompt(b01, b02, b03))
        if result is None:
            log.warning(f"Failed for {school}/{cls}/{pid}")
            continue

        features = encode_features(result, len(b01), len(b02), len(b03))
        np.save(out_path, features)
        success += 1
        time.sleep(delay)

    log.info(f"{split}: {success}/{len(participants)} participants processed")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    process_manifest(Path(args.manifest), Path(args.feature_root),
                     args.split, Path(args.output_dir),
                     limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
