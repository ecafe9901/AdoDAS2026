"""Transcript text features via learned character embeddings (no external models)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Standard A01 reading passage
_A01_TEXT = (
    "有一回，北风跟太阳在那儿争论谁的本领大。说着说着，来了一个过路的，"
    "身上穿了一件厚袍子。他们俩就商量好了，说谁能先叫这个过路的把他的袍子脱下来，"
    "就算是他的本领大。北风就使劲吹起来，拼命地吹。可是，他吹得越厉害，"
    "那个人就把他的袍子裹得越紧。到末了儿，北风没辙了，只好就算了。"
    "一会儿，太阳出来一晒，那个人马上就把袍子脱了下来。"
    "所以，北风不得不承认，还是太阳比他的本领大。"
)

DEFAULT_CHAR_DIM = 128
DEFAULT_TEXT_DIM = 128


def _build_char_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    """Build character vocabulary from transcript texts."""
    from collections import Counter
    counts = Counter()
    for t in texts:
        if t:
            counts.update(t)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, cnt in counts.most_common():
        if cnt >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def _char_ids(text: str, vocab: dict[str, int], max_len: int = 128) -> list[int]:
    if not text:
        return [0]
    return [vocab.get(ch, 1) for ch in text[:max_len]]


class CharTextEncoder(nn.Module):
    """Learned character-level text encoder for Chinese transcripts."""

    def __init__(self, vocab: dict[str, int], char_dim: int = 128,
                 text_dim: int = 128) -> None:
        super().__init__()
        self.vocab = vocab
        self.char_dim = char_dim
        self.text_dim = text_dim
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(char_dim, text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, char_ids: list[list[int]]) -> torch.Tensor:
        """char_ids: list of character-id lists (variable length)"""
        device = self.embedding.weight.device
        max_len = max(len(ids) for ids in char_ids)
        padded = torch.zeros(len(char_ids), max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(char_ids):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            padded[i, :len(ids)] = ids_t
        mask = (padded != 0).float().unsqueeze(-1)
        emb = self.embedding(padded)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.proj(pooled)


def load_transcript(root: Path, split: str, school: str, cls: str, pid: str,
                    session: str) -> str | None:
    """Load clean transcript text for a session."""
    if session == "A01":
        return _A01_TEXT
    path = root / split / school / cls / pid / session / "clean_transcript.txt"
    if not path.exists():
        return None
    text = path.read_text().strip()
    return text or None


def build_char_encoder_from_dataset(texts: list[str],
                                     char_dim: int = 128,
                                     text_dim: int = 128) -> CharTextEncoder:
    """Build a CharTextEncoder with vocabulary from collected texts."""
    vocab = _build_char_vocab(texts, min_freq=2)
    log.info(f"Char vocab size: {len(vocab)}")
    return CharTextEncoder(vocab, char_dim=char_dim, text_dim=text_dim)


def collect_all_transcripts(root: Path, split: str,
                             participants: list[dict]) -> list[str]:
    """Collect all transcript texts for vocabulary building."""
    texts = []
    for info in participants:
        for sess in ["A01", "B01", "B02", "B03"]:
            text = load_transcript(root, split,
                                   info["anon_school"], info["anon_class"],
                                   info["anon_pid"], sess)
            if text:
                texts.append(text)
    return texts


def get_text_dim() -> int:
    return DEFAULT_TEXT_DIM
