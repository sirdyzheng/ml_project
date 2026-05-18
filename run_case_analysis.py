#!/usr/bin/env python3
"""Generate case study analysis — find success and failure cases."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data, sample_data, extract_sessions, get_gold_turns
from src.chunking import chunk_turns
from src.embedder import encode_texts
from src.pipeline import run_flat, run_smc_qa
from src.evaluate import hit_at_k, recall_at_k, contains_match
from src.config import RESULTS_DIR


def main():
    print("案例分析：找出 SMC-QA 比 Flat Memory 好/差的典型例子\n")

    raw_data = load_raw_data()
    dev_data, test_data = sample_data(raw_data)

    cases = []
    all_texts = []
    text_ranges = []

    for item in test_data[:80]:
        turns = extract_sessions(item)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([c["text"] for c in chunks])
        all_texts.append(item["question"])
        text_ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"对 {len(text_ranges)} 条数据生成向量...")
    all_embs = encode_texts(all_texts, batch_size=128, show_progress=True)

    for start, query_idx, chunks, item in text_ranges:
        for i, c in enumerate(chunks):
            c["embedding"] = all_embs[start + i]
        query_emb = all_embs[query_idx]
        gold_texts = get_gold_turns(item)

        flat_results = run_flat(chunks, query_emb, query_text=item["question"])
        smc_results = run_smc_qa(chunks, query_emb, query_text=item["question"])

        flat_hit = hit_at_k(flat_results, gold_texts)
        smc_hit = hit_at_k(smc_results, gold_texts)
        flat_recall = recall_at_k(flat_results, gold_texts)
        smc_recall = recall_at_k(smc_results, gold_texts)

        cases.append({
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answer": str(item["answer"]),
            "num_chunks": len(chunks),
            "num_gold": len(gold_texts),
            "flat_hit": flat_hit,
            "flat_recall": flat_recall,
            "smc_hit": smc_hit,
            "smc_recall": smc_recall,
            "delta_recall": smc_recall - flat_recall,
            "flat_retrieved": [it.text[:100] for it in flat_results[:3]],
            "smc_retrieved": [it.text[:100] for it in smc_results[:3]],
        })

    # Sort by delta_recall
    cases.sort(key=lambda x: x["delta_recall"], reverse=True)

    print("\n" + "=" * 80)
    print("  SMC-QA 成功案例（比 Flat Memory 好）")
    print("=" * 80)
    for c in cases[:3]:
        print(f"\n  问题类型: {c['question_type']}")
        print(f"  问题: {c['question']}")
        print(f"  答案: {c['answer']}")
        print(f"  Flat Recall: {c['flat_recall']:.2f} → SMC Recall: {c['smc_recall']:.2f} (+{c['delta_recall']:.2f})")
        print(f"  SMC 找到的前3条:")
        for i, t in enumerate(c["smc_retrieved"]):
            print(f"    [{i+1}] {t}...")

    print("\n" + "=" * 80)
    print("  SMC-QA 失败案例（比 Flat Memory 差）")
    print("=" * 80)
    for c in cases[-3:]:
        print(f"\n  问题类型: {c['question_type']}")
        print(f"  问题: {c['question']}")
        print(f"  答案: {c['answer']}")
        print(f"  Flat Recall: {c['flat_recall']:.2f} → SMC Recall: {c['smc_recall']:.2f} ({c['delta_recall']:.2f})")
        print(f"  Flat 找到的前3条:")
        for i, t in enumerate(c["flat_retrieved"]):
            print(f"    [{i+1}] {t}...")

    # Save all cases
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "case_analysis.json", "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\n案例分析已保存到 results/case_analysis.json ({len(cases)} 条)")


if __name__ == "__main__":
    main()
