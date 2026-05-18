"""Main experiment pipeline — run all methods on a dataset split."""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from collections import defaultdict

from .config import (
    CONSOLIDATION_FREQ, PROMOTION_TOP_K, SCORE_THRESHOLD,
    SCORE_WEIGHTS, RETRIEVAL_TOP_K, STM_TOP_K, LTM_TOP_K,
    STM_CAPACITY, LTM_MAX_SIZE, RESULTS_DIR,
)
from .data_loader import load_raw_data, sample_data, extract_sessions, get_gold_turns
from .chunking import chunk_turns
from .embedder import encode_texts
from .memory import FlatMemory, STM, LTM, MemoryItem
from .promotion import promote, frequency_promote
from .router import retrieve_with_routing
from .evaluate import hit_at_k, recall_at_k, contains_match


# ---------------------------------------------------------------------------
# Method runners — each takes (chunks_with_emb, query_text, query_emb, gold_texts)
# and returns retrieved MemoryItems
# ---------------------------------------------------------------------------

def run_flat(chunks: List[Dict], query_emb: np.ndarray, **kw) -> List[Dict]:
    mem = FlatMemory()
    for c in chunks:
        mem.add(MemoryItem(c["chunk_id"], c["text"], c["embedding"],
                           c["session_id"], c["has_answer"]))
    return mem.retrieve(query_emb, RETRIEVAL_TOP_K)


def run_stm_only(chunks: List[Dict], query_emb: np.ndarray, **kw) -> List[Dict]:
    stm = STM(STM_CAPACITY)
    for c in chunks:
        stm.add(MemoryItem(c["chunk_id"], c["text"], c["embedding"],
                            c["session_id"], c["has_answer"]))
    return stm.retrieve(query_emb, STM_TOP_K)


def _build_memory(chunks, query_emb, promotion_fn="smc", threshold=SCORE_THRESHOLD,
                  weights=SCORE_WEIGHTS):
    """Shared helper: stream chunks into STM, periodically consolidate into LTM.
    Uses query_emb + recent chunk embeddings as proxy query signals."""
    stm = STM(STM_CAPACITY)
    ltm = LTM(LTM_MAX_SIZE)
    recent_embs = [query_emb]  # seed with actual question

    for i, c in enumerate(chunks):
        item = MemoryItem(c["chunk_id"], c["text"], c["embedding"],
                          c["session_id"], c["has_answer"])
        stm.add(item)
        # keep a rolling window of recent embeddings as "query signal"
        recent_embs.append(c["embedding"])
        if len(recent_embs) > 5:
            recent_embs.pop(0)

        # periodically retrieve to accumulate hit_count
        if (i + 1) % 3 == 0 and stm.items:
            stm.retrieve(query_emb, 2)

        if (i + 1) % CONSOLIDATION_FREQ == 0:
            if promotion_fn == "smc":
                promote(stm, ltm, recent_embs,
                        top_k=PROMOTION_TOP_K, threshold=threshold, weights=weights)
            elif promotion_fn == "freq":
                frequency_promote(stm, ltm, PROMOTION_TOP_K)
    # final consolidation for remaining items
    if promotion_fn == "smc":
        promote(stm, ltm, recent_embs, top_k=PROMOTION_TOP_K,
                threshold=threshold, weights=weights)
    elif promotion_fn == "freq":
        frequency_promote(stm, ltm, PROMOTION_TOP_K)
    return stm, ltm


def run_ltm_only(chunks: List[Dict], query_emb: np.ndarray,
                 query_text: str = "", **kw) -> List[Dict]:
    _, ltm = _build_memory(chunks, query_emb, "smc")
    return ltm.retrieve(query_emb, LTM_TOP_K)


def run_naive_stm_ltm(chunks: List[Dict], query_emb: np.ndarray,
                      query_text: str = "", **kw) -> List[Dict]:
    stm, ltm = _build_memory(chunks, query_emb, "smc")
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="hybrid")


def run_freq_promotion(chunks: List[Dict], query_emb: np.ndarray,
                       query_text: str = "", **kw) -> List[Dict]:
    stm, ltm = _build_memory(chunks, query_emb, "freq")
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="hybrid")


def run_smc_qa(chunks: List[Dict], query_emb: np.ndarray,
               query_text: str = "", **kw) -> List[Dict]:
    stm, ltm = _build_memory(chunks, query_emb, "smc")
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="auto")


# Ablation variants
def run_ablation_no_relevance(chunks, query_emb, query_text="", **kw):
    w = {"relevance": 0.0, "reuse": 0.6, "diversity": 0.4}
    stm, ltm = _build_memory(chunks, query_emb, "smc", weights=w)
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="auto")


def run_ablation_no_diversity(chunks, query_emb, query_text="", **kw):
    w = {"relevance": 0.5, "reuse": 0.5, "diversity": 0.0}
    stm, ltm = _build_memory(chunks, query_emb, "smc", weights=w)
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="auto")


def run_ablation_no_routing(chunks, query_emb, query_text="", **kw):
    stm, ltm = _build_memory(chunks, query_emb, "smc")
    return retrieve_with_routing(query_text, query_emb, stm, ltm, route="hybrid")


METHODS = {
    "Flat Memory": run_flat,
    "STM-only": run_stm_only,
    "LTM-only": run_ltm_only,
    "Naive STM+LTM": run_naive_stm_ltm,
    "Freq Promotion": run_freq_promotion,
    "SMC-QA (Ours)": run_smc_qa,
}

ABLATION_METHODS = {
    "Ablation: no relevance": run_ablation_no_relevance,
    "Ablation: no diversity": run_ablation_no_diversity,
    "Ablation: no routing": run_ablation_no_routing,
}


def run_experiment(data_items: List[Dict], methods: Dict = None,
                   label: str = "experiment", seed: int = 42) -> Dict[str, Dict]:
    if methods is None:
        methods = METHODS
    results = {name: {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
               for name in methods}

    print(f"\n[{label}] 开始实验 — {len(data_items)} 条数据, {len(methods)} 种方法")
    print(f"  第一步: 预处理所有数据（切块 + 向量化）...")

    preprocessed = []
    all_texts = []
    text_ranges = []
    for item in tqdm(data_items, desc="切块中"):
        turns = extract_sessions(item)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([c["text"] for c in chunks])
        all_texts.append(item["question"])
        text_ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"  共 {len(all_texts)} 段文本，正在生成向量...")
    all_embs = encode_texts(all_texts, batch_size=128, show_progress=True)

    for start, query_idx, chunks, item in text_ranges:
        for i, c in enumerate(chunks):
            c["embedding"] = all_embs[start + i]
        query_emb = all_embs[query_idx]
        gold_texts = get_gold_turns(item)
        preprocessed.append((chunks, item["question"], query_emb, gold_texts, item["answer"]))

    print(f"  预处理完成！开始逐方法评测...\n")

    for method_name, method_fn in methods.items():
        print(f"  方法: {method_name}")
        for chunks, query_text, query_emb, gold_texts, gold_answer in tqdm(
                preprocessed, desc=f"  {method_name}", leave=False):
            t0 = time.time()
            retrieved = method_fn(chunks, query_emb, query_text=query_text)
            latency = time.time() - t0

            h = hit_at_k(retrieved, gold_texts)
            r = recall_at_k(retrieved, gold_texts)

            pred_text = " ".join([it.text for it in retrieved[:3]])
            c = contains_match(pred_text, gold_answer)

            results[method_name]["hit@6"].append(h)
            results[method_name]["recall@6"].append(r)
            results[method_name]["contains"].append(c)
            results[method_name]["latency"].append(latency)

        avg = {k: np.mean(v) for k, v in results[method_name].items()}
        print(f"    Hit@6={avg['hit@6']:.3f}  Recall@6={avg['recall@6']:.3f}  "
              f"Contains={avg['contains']:.3f}  Latency={avg['latency']*1000:.1f}ms")

    return results


def summarize_results(all_results: Dict) -> Dict:
    summary = {}
    for method, metrics in all_results.items():
        summary[method] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                           for k, v in metrics.items()}
    return summary


def save_results(results: Dict, filename: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    serializable = {}
    for method, metrics in results.items():
        serializable[method] = {k: [float(x) for x in v] for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {path}")
