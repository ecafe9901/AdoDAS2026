#!/usr/bin/env python3
"""V1 LLM feature extraction — matches training/val encoding exactly.

Output: 34 dims = 21 DASS (0-3) + valence(1) + emotion_onehot(6)
                   + engagement_onehot(3) + richness_onehot(3)
"""

from __future__ import annotations

import argparse, json, logging, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import urllib3
from tqdm import tqdm

urllib3.disable_warnings()
log = logging.getLogger("extract_v1")

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = "deepseek-chat"
DASS_ITEMS = [f"d{i:02d}" for i in range(1, 22)]

_SYSTEM = """You are a clinical psychologist specializing in adolescent mental health.
Analyze three speech transcripts from a Chinese adolescent:
- B01: "How was your day yesterday?"
- B02: "Happiest memory from the past week"
- B03: "Saddest memory from the past week"

Score each DASS-21 item 0-3. Also provide psychological markers.
Normal adolescent complaints (exams, arguments) → score 0-1. Only score 2-3 with clear evidence.
Return ONLY a JSON object. No explanation."""


def build_prompt(b01: str, b02: str, b03: str) -> str:
    return f"""B01: "{b01}"
B02: "{b02}"
B03: "{b03}"

Return JSON:
{{"d01":0,"d02":0,...,"d21":0,
 "valence":"positive"/"negative"/"neutral",
 "primary_emotion":"joy"/"sadness"/"fear"/"anger"/"surprise"/"neutral",
 "engagement":"engaged"/"withdrawn"/"neutral",
 "richness":"detailed"/"minimal"/"avoidant"}}"""


def call_api(system: str, user: str) -> dict | None:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": user}], "max_tokens": 400, "temperature": 0.0}
    for _ in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=30, verify=False)
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"]["content"].strip()
                if c.startswith("```"): c = "\n".join(c.split("\n")[1:-1])
                return json.loads(c)
            elif r.status_code == 429: time.sleep(2)
            else: time.sleep(0.5)
        except: time.sleep(1)
    return None


def encode(result: dict) -> np.ndarray:
    """Encode to 34-dim vector: 21 DASS + 13 markers."""
    valence_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    emotion_map = {"joy": 0, "sadness": 1, "fear": 2, "anger": 3,
                   "surprise": 4, "neutral": 5}
    engagement_map = {"engaged": 0, "withdrawn": 1, "neutral": 2}
    richness_map = {"detailed": 0, "minimal": 1, "avoidant": 2}

    dass = np.clip([float(result.get(k, 0)) for k in DASS_ITEMS], 0, 3)
    valence = valence_map.get(result.get("valence", "neutral"), 0.0)

    emo_idx = emotion_map.get(result.get("primary_emotion", "neutral"), 5)
    emo_onehot = np.zeros(6, dtype=np.float32); emo_onehot[emo_idx] = 1.0

    eng_idx = engagement_map.get(result.get("engagement", "neutral"), 2)
    eng_onehot = np.zeros(3, dtype=np.float32); eng_onehot[eng_idx] = 1.0

    rich_idx = richness_map.get(result.get("richness", "minimal"), 1)
    rich_onehot = np.zeros(3, dtype=np.float32); rich_onehot[rich_idx] = 1.0

    return np.concatenate([dass, [valence], emo_onehot, eng_onehot, rich_onehot])


def extract_one(school, cls, pid, b01, b02, b03, out_path):
    if out_path.exists(): return True
    r = call_api(_SYSTEM, build_prompt(b01, b02, b03))
    if r is None: return False
    np.save(out_path, encode(r))
    return True


def process(manifest_path, feature_root, split, output_dir, limit=0, workers=5):
    df = pd.read_csv(manifest_path)
    grouped = df.groupby(["anon_school", "anon_class", "anon_pid"])
    participants = [(str(s), str(c), str(p)) for (s, c, p), _ in grouped]
    if limit: participants = participants[:limit]

    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_root = Path(feature_root)

    tasks = []
    for school, cls, pid in participants:
        out_path = out_dir / f"{school}_{cls}_{pid}.npy"
        if out_path.exists(): continue
        b01 = b02 = b03 = ""
        for sess in ["B01", "B02", "B03"]:
            p = feat_root / split / school / cls / pid / sess / "clean_transcript.txt"
            if p.exists():
                txt = p.read_text().strip()
                if sess == "B01": b01 = txt
                elif sess == "B02": b02 = txt
                else: b03 = txt
        tasks.append((school, cls, pid, b01, b02, b03, out_path))

    log.info(f"{split}: {len(tasks)} participants")
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_one, *t): t for t in tasks}
        for f in tqdm(futures, desc=f"Extract {split}", total=len(tasks)):
            try:
                if f.result(): ok += 1
            except: pass
    log.info(f"{split}: {ok}/{len(tasks)} done")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--split", default="test_hidden")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process(args.manifest, args.feature_root, args.split, Path(args.output_dir),
            limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
