#!/usr/bin/env python3
"""Pre-compute LLM-based features from transcripts using DeepSeek API.

Extracts per-participant:
  - DASS-21 direct scores (21 values) from the LLM's clinical assessment
  - Psychological markers: valence, emotion, engagement, richness

Stores results as .npy arrays alongside existing pooled features.

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
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-8af11503cf7240e582434134d37c44ce")
MODEL = "deepseek-chat"

DASS_ITEMS = [f"d{i:02d}" for i in range(1, 22)]

_SYSTEM_PROMPT = """You are a clinical psychologist specializing in adolescent mental health assessment.
Analyze the following three speech transcripts from a Chinese adolescent.

The three responses are:
- B01: "How was your day yesterday?" (请描述一下，你昨天过的怎么样?)
- B02: "Happiest memory from the past week" (请描述一下，现在回想最近一周最开心的记忆?)
- B03: "Saddest memory from the past week" (请描述一下，现在回想最近一周最悲伤的记忆?)

Assess the adolescent on the DASS-21 (Depression Anxiety Stress Scales) items.
Score each item 0-3: 0=not at all, 1=mild/sometimes, 2=moderate/often, 3=severe/almost always.

Also provide psychological markers:
- valence: "positive", "negative", or "neutral"
- primary_emotion: "joy", "sadness", "fear", "anger", "surprise", or "neutral"
- engagement: "engaged", "withdrawn", or "neutral"
- richness: "detailed", "minimal", or "avoidant"

Be objective. Even if transcripts are short, make your best assessment based on available information.
Return ONLY a JSON object. No explanations."""


def build_user_prompt(b01_text: str, b02_text: str, b03_text: str) -> str:
    return f"""B01 (How was your day?): "{b01_text}"
B02 (Happiest memory): "{b02_text}"
B03 (Saddest memory): "{b03_text}"

Return JSON with keys d01-d21 (int 0-3), valence (str), primary_emotion (str),
engagement (str), richness (str):"""


def call_deepseek(system: str, user: str, max_retries: int = 3) -> dict | None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 500,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=headers, json=data,
                              timeout=60, verify=False)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                # Extract JSON from response (may have markdown wrapping)
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1])
                return json.loads(content)
            elif r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"API error {r.status_code}: {r.text[:200]}")
                time.sleep(1)
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def encode_markers(result: dict) -> np.ndarray:
    """Encode psychological markers as a fixed-size float vector."""
    valence_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    emotion_map = {"joy": 0, "sadness": 1, "fear": 2, "anger": 3,
                   "surprise": 4, "neutral": 5}
    engagement_map = {"engaged": 0, "withdrawn": 1, "neutral": 2}
    richness_map = {"detailed": 0, "minimal": 1, "avoidant": 2}

    features = []
    # Valence (1)
    features.append(valence_map.get(result.get("valence", "neutral"), 0.0))
    # Emotion one-hot (6)
    emo = emotion_map.get(result.get("primary_emotion", "neutral"), 5)
    emo_onehot = np.zeros(6, dtype=np.float32)
    emo_onehot[emo] = 1.0
    features.extend(emo_onehot)
    # Engagement one-hot (3)
    eng = engagement_map.get(result.get("engagement", "neutral"), 2)
    eng_onehot = np.zeros(3, dtype=np.float32)
    eng_onehot[eng] = 1.0
    features.extend(eng_onehot)
    # Richness one-hot (3)
    rich = richness_map.get(result.get("richness", "minimal"), 1)
    rich_onehot = np.zeros(3, dtype=np.float32)
    rich_onehot[rich] = 1.0
    features.extend(rich_onehot)
    return np.array(features, dtype=np.float32)  # 1 + 6 + 3 + 3 = 13 dims


def extract_features(b01_text: str, b02_text: str, b03_text: str) -> dict | None:
    """Extract LLM features for one participant. Returns None on failure."""
    user = build_user_prompt(b01_text, b02_text, b03_text)
    result = call_deepseek(_SYSTEM_PROMPT, user)
    return result


def load_transcript(feature_root: Path, split: str, school: str, cls: str,
                    pid: str, session: str) -> str:
    path = feature_root / split / school / cls / pid / session / "clean_transcript.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


def process_manifest(manifest_path: Path, feature_root: Path, split: str,
                     output_dir: Path, limit: int = 0,
                     delay: float = 0.5) -> int:
    """Process all participants in a manifest CSV."""
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

        if not any([b01, b02, b03]):
            continue

        result = extract_features(b01, b02, b03)
        if result is None:
            log.warning(f"Failed for {school}/{cls}/{pid}")
            continue

        # Encode: 21 DASS scores (float) + 13 marker features (float) = 34 dims
        dass_scores = np.array([float(result.get(k, 0)) for k in DASS_ITEMS],
                               dtype=np.float32)
        markers = encode_markers(result)
        features = np.concatenate([dass_scores, markers])  # (34,)

        np.save(out_path, features)
        success += 1
        time.sleep(delay)  # rate limiting

    log.info(f"{split}: {success}/{len(participants)} participants processed")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit participants (for testing)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    process_manifest(
        Path(args.manifest), Path(args.feature_root),
        args.split, Path(args.output_dir),
        limit=args.limit, delay=args.delay,
    )


if __name__ == "__main__":
    main()
