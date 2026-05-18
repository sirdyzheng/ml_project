#!/usr/bin/env python3
"""Run experiments on s_cleaned data (full haystack with distractors)."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data, sample_data, extract_sessions, get_gold_turns
from src.chunking import chunk_turns
from src.embedder import encode_texts
from src.memory import FlatMemory, STM, LTM, MemoryItem
from src.pipeline import (
    METHODS, ABLATION_METHODS, save_results, summarize_results,
    run_flat, run_stm_only, run_ltm_only, run_naive_stm_ltm,
    run_freq_promotion, run_smc_qa,
)
from src.evaluate import hit_at_k, recall_at_k, contains_match
from src.config import RESULTS_DIR, RETRIEVAL_TOP_K
from tqdm import tqdm
import time

MAX_TURNS = 200


def run_s_cleaned_experiment():
    print("=" * 60)
    print("  SMC-QA — S-Cleaned 数据（大海捞针场景）")
    print("=" * 60)

    print("\n加载 s_cleaned 数据...")
    raw_data = load_raw_data("longmemeval_s_cleaned.json")
    print(f"  总条数: {len(raw_data)}")

    # Take a sample of 100 items for tractable runtime
    import random
    random.seed(42)
    indices = list(range(len(raw_data)))
    random.shuffle(indices)
    data_items = [raw_data[i] for i in indices[:100]]
    print(f"  抽样: 100 条 (seed=42)")
    print(f"  每条截断至 {MAX_TURNS} turns")

    methods = METHODS

    results = {name: {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
               for name in methods}

    print(f"\n预处理（切块+向量化）...")
    all_texts = []
    text_ranges = []
    for item in tqdm(data_items, desc="切块中"):
        turns = extract_sessions(item, max_turns=MAX_TURNS)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([c["text"] for c in chunks])
        all_texts.append(item["question"])
        text_ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"  共 {len(all_texts)} 段文本，生成向量...")
    all_embs = encode_texts(all_texts, batch_size=128, show_progress=True)

    preprocessed = []
    for start, query_idx, chunks, item in text_ranges:
        for i, c in enumerate(chunks):
            c["embedding"] = all_embs[start + i]
        query_emb = all_embs[query_idx]
        gold_texts = get_gold_turns(item)
        preprocessed.append((chunks, item["question"], query_emb, gold_texts, item["answer"]))

    print(f"\n开始逐方法评测...\n")
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
            c = contains_match(pred_text, str(gold_answer))
            results[method_name]["hit@6"].append(h)
            results[method_name]["recall@6"].append(r)
            results[method_name]["contains"].append(c)
            results[method_name]["latency"].append(latency)

        avg = {k: np.mean(v) for k, v in results[method_name].items()}
        print(f"    Hit@6={avg['hit@6']:.3f}  Recall@6={avg['recall@6']:.3f}  "
              f"Contains={avg['contains']:.3f}  Latency={avg['latency']*1000:.1f}ms")

    # Print final table
    print(f"\n{'='*80}")
    print(f"  S-Cleaned 结果 (100条, max_turns={MAX_TURNS})")
    print(f"{'='*80}")
    header = f"{'Method':<25} {'Hit@6':>10} {'Recall@6':>10} {'Contains':>10}"
    print(header)
    print("-" * 60)
    for method, metrics in results.items():
        h = np.mean(metrics["hit@6"])
        r = np.mean(metrics["recall@6"])
        c = np.mean(metrics["contains"])
        print(f"{method:<25} {h:.3f}      {r:.3f}      {c:.3f}")
    print("=" * 80)

    save_results(results, "s_cleaned_results.json")


if __name__ == "__main__":
    run_s_cleaned_experiment()
