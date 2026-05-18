"""STM / LTM memory structures and Flat Memory."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .config import STM_CAPACITY, LTM_MAX_SIZE, DUPLICATE_SIM_THRESHOLD
from .embedder import cosine_sim


@dataclass
class MemoryItem:
    chunk_id: int
    text: str
    embedding: np.ndarray
    session_id: str
    has_answer: bool
    hit_count: int = 0
    age: int = 0

    def __hash__(self):
        return self.chunk_id

    def __eq__(self, other):
        return self.chunk_id == other.chunk_id


class FlatMemory:
    """All chunks in one pool — simplest baseline."""

    def __init__(self):
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        self.items.append(item)

    def retrieve(self, query_emb: np.ndarray, top_k: int = 6) -> List[MemoryItem]:
        if not self.items:
            return []
        embs = np.stack([it.embedding for it in self.items])
        sims = cosine_sim(query_emb, embs).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [self.items[i] for i in top_indices]
        for it in results:
            it.hit_count += 1
        return results

    def all_embeddings(self) -> Optional[np.ndarray]:
        if not self.items:
            return None
        return np.stack([it.embedding for it in self.items])


class STM:
    """Short-term memory: FIFO with fixed capacity."""

    def __init__(self, capacity: int = STM_CAPACITY):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        self.items.append(item)
        while len(self.items) > self.capacity:
            self.items.pop(0)

    def retrieve(self, query_emb: np.ndarray, top_k: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        embs = np.stack([it.embedding for it in self.items])
        sims = cosine_sim(query_emb, embs).flatten()
        k = min(top_k, len(self.items))
        top_indices = np.argsort(sims)[::-1][:k]
        results = [self.items[i] for i in top_indices]
        for it in results:
            it.hit_count += 1
        return results

    def all_embeddings(self) -> Optional[np.ndarray]:
        if not self.items:
            return None
        return np.stack([it.embedding for it in self.items])

    def avg_similarity(self, query_emb: np.ndarray) -> float:
        if not self.items:
            return 0.0
        embs = self.all_embeddings()
        return float(cosine_sim(query_emb, embs).mean())


class LTM:
    """Long-term memory: only promoted items live here."""

    def __init__(self, max_size: int = LTM_MAX_SIZE):
        self.max_size = max_size
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> bool:
        if self.items:
            embs = np.stack([it.embedding for it in self.items])
            sims = cosine_sim(item.embedding, embs).flatten()
            if sims.max() >= DUPLICATE_SIM_THRESHOLD:
                return False
        self.items.append(item)
        self._evict_if_needed()
        return True

    def _evict_if_needed(self):
        while len(self.items) > self.max_size:
            worst = min(self.items, key=lambda it: it.hit_count)
            self.items.remove(worst)

    def retrieve(self, query_emb: np.ndarray, top_k: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        embs = np.stack([it.embedding for it in self.items])
        sims = cosine_sim(query_emb, embs).flatten()
        k = min(top_k, len(self.items))
        top_indices = np.argsort(sims)[::-1][:k]
        results = [self.items[i] for i in top_indices]
        for it in results:
            it.hit_count += 1
        return results

    def all_embeddings(self) -> Optional[np.ndarray]:
        if not self.items:
            return None
        return np.stack([it.embedding for it in self.items])

    def avg_similarity(self, query_emb: np.ndarray) -> float:
        if not self.items:
            return 0.0
        embs = self.all_embeddings()
        return float(cosine_sim(query_emb, embs).mean())

    def redundancy_rate(self) -> float:
        if len(self.items) < 2:
            return 0.0
        embs = np.stack([it.embedding for it in self.items])
        sims = cosine_sim(embs, embs)
        np.fill_diagonal(sims, 0)
        n = len(self.items)
        total_pairs = n * (n - 1) / 2
        dup_count = (sims >= DUPLICATE_SIM_THRESHOLD).sum() / 2
        return float(dup_count / total_pairs) if total_pairs > 0 else 0.0
