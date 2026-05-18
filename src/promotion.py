"""Promotion: decide which STM items get promoted to LTM."""

import numpy as np
from typing import List
from .memory import MemoryItem, STM, LTM
from .embedder import cosine_sim
from .config import (
    SCORE_WEIGHTS, SCORE_THRESHOLD, PROMOTION_TOP_K,
    CONSOLIDATION_FREQ,
)


def compute_relevance(item: MemoryItem, recent_query_embs: List[np.ndarray]) -> float:
    if not recent_query_embs:
        return 0.0
    sims = [float(cosine_sim(item.embedding, q).item()) for q in recent_query_embs]
    return np.mean(sims)


def compute_reuse(item: MemoryItem, max_possible: int) -> float:
    if max_possible <= 0:
        return 0.0
    return min(item.hit_count / max_possible, 1.0)


def compute_diversity(item: MemoryItem, ltm: LTM) -> float:
    if not ltm.items:
        return 1.0
    ltm_embs = ltm.all_embeddings()
    sims = cosine_sim(item.embedding, ltm_embs).flatten()
    return float(1.0 - sims.max())


def score_item(item: MemoryItem, recent_query_embs: List[np.ndarray],
               ltm: LTM, max_reuse: int,
               weights: dict = SCORE_WEIGHTS) -> float:
    rel = compute_relevance(item, recent_query_embs)
    reu = compute_reuse(item, max_reuse)
    div = compute_diversity(item, ltm)
    return weights["relevance"] * rel + weights["reuse"] * reu + weights["diversity"] * div


def promote(stm: STM, ltm: LTM, recent_query_embs: List[np.ndarray],
            top_k: int = PROMOTION_TOP_K, threshold: float = SCORE_THRESHOLD,
            weights: dict = SCORE_WEIGHTS) -> int:
    """Score STM items, promote top-k above threshold into LTM. Returns count promoted."""
    if not stm.items:
        return 0
    max_reuse = max(it.hit_count for it in stm.items) if stm.items else 1
    scored = []
    for item in stm.items:
        s = score_item(item, recent_query_embs, ltm, max_reuse, weights)
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    promoted = 0
    for s, item in scored[:top_k]:
        if s >= threshold:
            if ltm.add(item):
                promoted += 1
    return promoted


def frequency_promote(stm: STM, ltm: LTM, top_k: int = PROMOTION_TOP_K) -> int:
    """Baseline: promote by hit count only."""
    if not stm.items:
        return 0
    ranked = sorted(stm.items, key=lambda it: it.hit_count, reverse=True)
    promoted = 0
    for item in ranked[:top_k]:
        if ltm.add(item):
            promoted += 1
    return promoted
