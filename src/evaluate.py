"""Evaluation metrics: Hit@k, Recall@k, Answer Accuracy."""

import re
import string
from typing import List
from .memory import MemoryItem


def normalize_text(text) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_hits_gold(retrieved: List[MemoryItem], gold_texts: List[str], threshold: float = 0.5) -> List[bool]:
    """For each gold text, check if any retrieved chunk contains enough overlap."""
    hits = []
    for gold in gold_texts:
        gold_words = set(normalize_text(gold).split())
        hit = False
        for item in retrieved:
            chunk_words = set(normalize_text(item.text).split())
            if not gold_words:
                continue
            overlap = len(gold_words & chunk_words) / len(gold_words)
            if overlap >= threshold:
                hit = True
                break
        hits.append(hit)
    return hits


def hit_at_k(retrieved: List[MemoryItem], gold_texts: List[str]) -> float:
    if not gold_texts:
        return 0.0
    hits = chunk_hits_gold(retrieved, gold_texts)
    return 1.0 if any(hits) else 0.0


def recall_at_k(retrieved: List[MemoryItem], gold_texts: List[str]) -> float:
    if not gold_texts:
        return 0.0
    hits = chunk_hits_gold(retrieved, gold_texts)
    return sum(hits) / len(hits)


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize_text(predicted) == normalize_text(gold) else 0.0


def contains_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize_text(gold) in normalize_text(predicted) else 0.0
