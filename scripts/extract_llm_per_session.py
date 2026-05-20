#!/usr/bin/env python3
"""Per-session LLM feature extraction with cross-session consistency features.

Calls DeepSeek API for EACH of B01/B02/B03 separately, then computes
cross-session comparison features. Stores in separate v3 directory.

Output: ~75 dims = 3×(21 DASS + 1 quality + 1 valence)
                  + 3×(DASS std per cluster)
                  + 2× emotional_range
                  + 1× gating
"""

from __future__ import annotations

import argparse, json, logging, os, time, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, urllib3, pandas as pd
from tqdm import tqdm

urllib3.disable_warnings()
log = logging.getLogger("extract_llm_v3")

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = "deepseek-chat"
DASS_ITEMS = [f"d{i:02d}" for i in range(1, 22)]

_SYSTEM = """You are a clinical psychologist. Assess DASS-21 from ONE speech transcript.
Normal adolescent complaints (exams, arguments) → score 0-1 only.
Short/garbled text → all zeros, quality=0.
Only score 2-3 with CLEAR clinical evidence.
Return ONLY JSON: d01-d21 (int 0-3), quality (float 0-1), valence (positive/negative/neutral)."""


def call_deepseek(system: str, user: str) -> dict | None:
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


def extract_one(school, cls, pid, b01, b02, b03, out_path):
    if out_path.exists():
        return True

    results = {}
    for sess_name, text in [("B01", b01), ("B02", b02), ("B03", b03)]:
        user = f'Session {sess_name}: "{text}"\nReturn JSON.'
        r = call_deepseek(_SYSTEM, user)
        if r is None:
            return False
        results[sess_name] = r

    # Encode per-session features
    per_session = []
    for sess in ["B01", "B02", "B03"]:
        r = results[sess]
        dass = np.clip([float(r.get(k, 0)) for k in DASS_ITEMS], 0, 3)
        quality = float(r.get("quality", 0.5))
        valence = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(
            r.get("valence", "neutral"), 0.0)
        per_session.append(np.concatenate([dass, [quality, valence]]))

    # Cross-session features
    dass_b01, dass_b02, dass_b03 = [p[:21] for p in per_session]
    dass_all = np.stack([dass_b01, dass_b02, dass_b03])  # (3, 21)
    dass_std = dass_all.std(axis=0)  # per-item inconsistency across sessions
    q_b01, q_b02, q_b03 = [p[21] for p in per_session]
    v_b01, v_b02, v_b03 = [p[22] for p in per_session]

    # Gating: if any session has terrible quality, downweight
    gating = min(q_b01, q_b02, q_b03)

    cross = np.array([
        float(np.mean(dass_std)),                     # avg inconsistency
        float(v_b02 - v_b03),                         # emotional range
        float(max(q_b01, q_b02, q_b03) - min(q_b01, q_b02, q_b03)),  # quality variance
        float(dass_b02.sum() - dass_b03.sum()),      # DASS B02 vs B03
        gating,
    ], dtype=np.float32)

    features = np.concatenate([
        per_session[0], per_session[1], per_session[2], cross
    ])  # 3×23 + 5 = 74 dims
    np.save(out_path, features)
    return True


def process_manifest(manifest_path, feature_root, split, output_dir,
                     limit=0, max_workers=5):
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

    log.info(f"{split}: {len(tasks)} participants to process")
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(extract_one, *t): t for t in tasks}
        for f in tqdm(futures, desc=f"Extract {split}", total=len(tasks)):
            if f.result(): success += 1
    log.info(f"{split}: {success}/{len(tasks)} done")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_manifest(args.manifest, args.feature_root, args.split,
                     Path(args.output_dir), limit=args.limit, max_workers=args.workers)


if __name__ == "__main__":
    main()
