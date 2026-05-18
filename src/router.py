"""Query-aware routing: decide whether to search STM, LTM, or both."""

import numpy as np
from .config import HISTORY_KEYWORDS, ROUTING_SIM_MARGIN, STM_TOP_K, LTM_TOP_K, RETRIEVAL_TOP_K
from .memory import STM, LTM, MemoryItem
from .embedder import cosine_sim
from typing import List


def route_query(query_text: str, query_emb: np.ndarray, stm: STM, ltm: LTM) -> str:
    query_lower = query_text.lower()
    for kw in HISTORY_KEYWORDS:
        if kw in query_lower:
            return "ltm"

    stm_sim = stm.avg_similarity(query_emb)
    ltm_sim = ltm.avg_similarity(query_emb)

    if stm_sim - ltm_sim >= ROUTING_SIM_MARGIN:
        return "stm"

    return "hybrid"


def retrieve_with_routing(query_text: str, query_emb: np.ndarray,
                          stm: STM, ltm: LTM,
                          route: str = "auto") -> List[MemoryItem]:
    if route == "auto":
        route = route_query(query_text, query_emb, stm, ltm)

    if route == "stm":
        return stm.retrieve(query_emb, STM_TOP_K)
    elif route == "ltm":
        return ltm.retrieve(query_emb, LTM_TOP_K)
    else:
        stm_results = stm.retrieve(query_emb, STM_TOP_K)
        ltm_results = ltm.retrieve(query_emb, LTM_TOP_K)
        seen_ids = set()
        merged = []
        for it in stm_results + ltm_results:
            if it.chunk_id not in seen_ids:
                seen_ids.add(it.chunk_id)
                merged.append(it)
        if len(merged) <= RETRIEVAL_TOP_K:
            return merged
        embs = np.stack([it.embedding for it in merged])
        sims = cosine_sim(query_emb, embs).flatten()
        top_indices = np.argsort(sims)[::-1][:RETRIEVAL_TOP_K]
        return [merged[i] for i in top_indices]
