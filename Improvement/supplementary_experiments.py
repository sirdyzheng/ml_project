#!/usr/bin/env python3
"""Supplementary analyses for the improvement study.

Outputs:
- question_type_breakdown_seed42.json
- ltm_quality_seed42.json
- tuned_case_studies_seed42.json
- retrieval_depth_ablation_summary.json
- supplementary_summary.md

All computations use the existing repository code without editing src/.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENT_DIR = PROJECT_ROOT / "Improvement"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IMPROVEMENT_DIR))

import src.router as router
from src import pipeline
from src.chunking import chunk_turns
from src.data_loader import extract_sessions, get_gold_turns, load_raw_data, sample_data
from src.embedder import encode_texts
from src.evaluate import contains_match, hit_at_k, recall_at_k
from tune_smc import temporary_pipeline_globals

RESULTS_DIR = PROJECT_ROOT / "Improvement" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TUNED_CONFIG = {
    "name": "Retrieval-tuned SMC-QA",
    "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2},
    "threshold": 0.40,
    "promotion_top_k": 3,
    "route": "auto",
    "stm_top_k": 8,
    "ltm_top_k": 8,
}


@contextmanager
def temporary_router_globals(**updates):
    old_values = {name: getattr(router, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(router, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(router, name, value)


def mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_metric_lists(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: mean(values) for key, values in metrics.items()}


def preprocess(data_items: List[Dict[str, Any]], label: str):
    print(f"\n[{label}] preprocessing {len(data_items)} items")
    all_texts: List[str] = []
    ranges = []
    for item in tqdm(data_items, desc=f"{label} chunking"):
        turns = extract_sessions(item)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([chunk["text"] for chunk in chunks])
        all_texts.append(item["question"])
        ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"[{label}] encoding {len(all_texts)} texts")
    all_embs = encode_texts(all_texts, batch_size=128, show_progress=True)

    preprocessed = []
    for start, query_idx, chunks, item in ranges:
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = all_embs[start + i]
        preprocessed.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "answer": str(item["answer"]),
                "chunks": chunks,
                "query_emb": all_embs[query_idx],
                "gold_texts": get_gold_turns(item),
            }
        )
    return preprocessed


def run_tuned_smc(sample: Dict[str, Any]):
    with temporary_pipeline_globals(PROMOTION_TOP_K=TUNED_CONFIG["promotion_top_k"]):
        stm, ltm = pipeline._build_memory(
            sample["chunks"],
            sample["query_emb"],
            promotion_fn="smc",
            threshold=TUNED_CONFIG["threshold"],
            weights=TUNED_CONFIG["weights"],
        )
    with temporary_router_globals(
        STM_TOP_K=TUNED_CONFIG["stm_top_k"],
        LTM_TOP_K=TUNED_CONFIG["ltm_top_k"],
    ):
        return router.retrieve_with_routing(
            sample["question"],
            sample["query_emb"],
            stm,
            ltm,
            route=TUNED_CONFIG["route"],
        )


def run_method(sample: Dict[str, Any], method_name: str):
    if method_name == "Flat Memory":
        return pipeline.run_flat(sample["chunks"], sample["query_emb"])
    if method_name == "Freq Promotion":
        return pipeline.run_freq_promotion(
            sample["chunks"],
            sample["query_emb"],
            query_text=sample["question"],
        )
    if method_name == "Original SMC-QA":
        return pipeline.run_smc_qa(
            sample["chunks"],
            sample["query_emb"],
            query_text=sample["question"],
        )
    if method_name == "Retrieval-tuned SMC-QA":
        return run_tuned_smc(sample)
    raise ValueError(f"Unknown method: {method_name}")


def eval_retrieved(retrieved, sample: Dict[str, Any]):
    pred_text = " ".join(item.text for item in retrieved[:3])
    return {
        "hit@6": hit_at_k(retrieved, sample["gold_texts"]),
        "recall@6": recall_at_k(retrieved, sample["gold_texts"]),
        "contains": contains_match(pred_text, sample["answer"]),
    }


def question_type_breakdown(preprocessed):
    methods = [
        "Flat Memory",
        "Freq Promotion",
        "Original SMC-QA",
        "Retrieval-tuned SMC-QA",
    ]
    by_type = {
        method: defaultdict(lambda: {"hit@6": [], "recall@6": [], "contains": [], "count": 0})
        for method in methods
    }
    per_sample = []

    print("\n[question type] evaluating methods")
    for sample in tqdm(preprocessed, desc="question type"):
        sample_row = {
            "question_id": sample["question_id"],
            "question_type": sample["question_type"],
            "question": sample["question"],
            "answer": sample["answer"],
            "methods": {},
        }
        for method in methods:
            t0 = time.time()
            retrieved = run_method(sample, method)
            metrics = eval_retrieved(retrieved, sample)
            metrics["latency"] = time.time() - t0
            sample_row["methods"][method] = {
                **metrics,
                "top3": [item.text[:240] for item in retrieved[:3]],
            }
            bucket = by_type[method][sample["question_type"]]
            bucket["count"] += 1
            for metric in ["hit@6", "recall@6", "contains"]:
                bucket[metric].append(metrics[metric])
        per_sample.append(sample_row)

    summarized = {}
    for method, type_map in by_type.items():
        summarized[method] = {}
        for qtype, metrics in sorted(type_map.items()):
            summarized[method][qtype] = {
                "count": metrics["count"],
                **summarize_metric_lists(
                    {k: v for k, v in metrics.items() if isinstance(v, list)}
                ),
            }
    return {"by_type": summarized, "per_sample": per_sample}


def build_memory_for_quality(sample: Dict[str, Any], variant: str):
    if variant == "Freq Promotion":
        return pipeline._build_memory(sample["chunks"], sample["query_emb"], "freq")[1]
    if variant == "Original SMC-QA":
        return pipeline._build_memory(sample["chunks"], sample["query_emb"], "smc")[1]
    if variant == "Retrieval-tuned SMC-QA":
        with temporary_pipeline_globals(PROMOTION_TOP_K=TUNED_CONFIG["promotion_top_k"]):
            return pipeline._build_memory(
                sample["chunks"],
                sample["query_emb"],
                "smc",
                threshold=TUNED_CONFIG["threshold"],
                weights=TUNED_CONFIG["weights"],
            )[1]
    raise ValueError(f"Unknown LTM variant: {variant}")


def ltm_quality(preprocessed):
    variants = ["Freq Promotion", "Original SMC-QA", "Retrieval-tuned SMC-QA"]
    rows = []
    print("\n[LTM quality] evaluating memory quality")
    for sample in tqdm(preprocessed, desc="ltm quality"):
        for variant in variants:
            ltm = build_memory_for_quality(sample, variant)
            ltm_items = list(ltm.items)
            answer_items = sum(1 for item in ltm_items if item.has_answer)
            rows.append(
                {
                    "question_id": sample["question_id"],
                    "question_type": sample["question_type"],
                    "variant": variant,
                    "ltm_size": len(ltm_items),
                    "answer_items": answer_items,
                    "answer_item_rate": answer_items / len(ltm_items) if ltm_items else 0.0,
                    "gold_hit_in_ltm": hit_at_k(ltm_items, sample["gold_texts"]),
                    "gold_recall_in_ltm": recall_at_k(ltm_items, sample["gold_texts"]),
                    "redundancy_rate": ltm.redundancy_rate(),
                }
            )

    overall = {}
    by_type = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        overall[variant] = {
            metric: mean([row[metric] for row in variant_rows])
            for metric in [
                "ltm_size",
                "answer_items",
                "answer_item_rate",
                "gold_hit_in_ltm",
                "gold_recall_in_ltm",
                "redundancy_rate",
            ]
        }
        by_type[variant] = {}
        for qtype in sorted({row["question_type"] for row in variant_rows}):
            subset = [row for row in variant_rows if row["question_type"] == qtype]
            by_type[variant][qtype] = {
                "count": len(subset),
                **{
                    metric: mean([row[metric] for row in subset])
                    for metric in [
                        "ltm_size",
                        "answer_item_rate",
                        "gold_hit_in_ltm",
                        "gold_recall_in_ltm",
                        "redundancy_rate",
                    ]
                },
            }
    return {"overall": overall, "by_type": by_type, "per_sample": rows}


def tuned_case_studies(question_type_results):
    cases = []
    for row in question_type_results["per_sample"]:
        original = row["methods"]["Original SMC-QA"]
        tuned = row["methods"]["Retrieval-tuned SMC-QA"]
        freq = row["methods"]["Freq Promotion"]
        cases.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
                "original_hit": original["hit@6"],
                "tuned_hit": tuned["hit@6"],
                "freq_hit": freq["hit@6"],
                "original_recall": original["recall@6"],
                "tuned_recall": tuned["recall@6"],
                "freq_recall": freq["recall@6"],
                "delta_recall_tuned_vs_original": tuned["recall@6"] - original["recall@6"],
                "delta_recall_tuned_vs_freq": tuned["recall@6"] - freq["recall@6"],
                "original_top3": original["top3"],
                "tuned_top3": tuned["top3"],
                "freq_top3": freq["top3"],
            }
        )

    improved = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_original"] > 0],
        key=lambda case: (case["delta_recall_tuned_vs_original"], case["tuned_hit"]),
        reverse=True,
    )
    regressed = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_original"] < 0],
        key=lambda case: case["delta_recall_tuned_vs_original"],
    )
    freq_better = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_freq"] < 0],
        key=lambda case: case["delta_recall_tuned_vs_freq"],
    )
    return {
        "improved_vs_original": improved[:10],
        "regressed_vs_original": regressed[:10],
        "freq_better_than_tuned": freq_better[:10],
    }


def retrieval_depth_ablation_summary():
    dev_path = RESULTS_DIR / "smc_retrieval_tuning_dev_results.json"
    test_path = RESULTS_DIR / "smc_retrieval_tuned_test_results.json"
    summary = {"dev_top10": [], "test_merged": {}}

    if dev_path.exists():
        with open(dev_path) as f:
            dev_rows = json.load(f)
        for row in dev_rows[:10]:
            summary["dev_top10"].append(
                {
                    "name": row["config"]["name"],
                    "stm_top_k": row["config"]["stm_top_k"],
                    "ltm_top_k": row["config"]["ltm_top_k"],
                    "hit@6": row["summary"]["hit@6"],
                    "recall@6": row["summary"]["recall@6"],
                    "contains": row["summary"]["contains"],
                    "objective": row["summary"]["objective"],
                }
            )

    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)
        summary["test_merged"] = test_data.get("merged", {})

    return summary


def write_markdown_summary(qtype, ltm, cases, depth):
    lines = [
        "# Supplementary Experiment Summary",
        "",
        "## 1. Question Type Breakdown",
        "",
        "Seed 42 test split; methods: Flat Memory, Freq Promotion, Original SMC-QA, Retrieval-tuned SMC-QA.",
        "",
    ]

    for method in ["Original SMC-QA", "Retrieval-tuned SMC-QA", "Freq Promotion"]:
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Question Type | Count | Hit@6 | Recall@6 | Contains |")
        lines.append("|---------------|------:|------:|----------:|---------:|")
        for qtype_name, metrics in qtype["by_type"][method].items():
            lines.append(
                f"| {qtype_name} | {metrics['count']} | "
                f"{metrics['hit@6']:.3f} | {metrics['recall@6']:.3f} | {metrics['contains']:.3f} |"
            )
        lines.append("")

    lines += [
        "## 2. LTM Quality",
        "",
        "| Variant | LTM Size | Answer Item Rate | Gold Hit In LTM | Gold Recall In LTM | Redundancy |",
        "|---------|---------:|-----------------:|----------------:|-------------------:|-----------:|",
    ]
    for variant, metrics in ltm["overall"].items():
        lines.append(
            f"| {variant} | {metrics['ltm_size']:.1f} | "
            f"{metrics['answer_item_rate']:.3f} | {metrics['gold_hit_in_ltm']:.3f} | "
            f"{metrics['gold_recall_in_ltm']:.3f} | {metrics['redundancy_rate']:.3f} |"
        )

    lines += [
        "",
        "## 3. Retrieval Depth Ablation",
        "",
        "Top dev configurations from retrieval-depth search:",
        "",
        "| Config | STM Top-K | LTM Top-K | Hit@6 | Recall@6 | Contains |",
        "|--------|----------:|----------:|------:|----------:|---------:|",
    ]
    for row in depth["dev_top10"][:5]:
        lines.append(
            f"| {row['name']} | {row['stm_top_k']} | {row['ltm_top_k']} | "
            f"{row['hit@6']:.3f} | {row['recall@6']:.3f} | {row['contains']:.3f} |"
        )

    lines += [
        "",
        "Merged test results for top retrieval-depth configs:",
        "",
        "| Config | Hit@6 | Recall@6 | Contains |",
        "|--------|------:|----------:|---------:|",
    ]
    for name, metrics in depth["test_merged"].items():
        lines.append(
            f"| {name} | {metrics['hit@6']['mean']:.3f} +/- {metrics['hit@6']['std']:.3f} | "
            f"{metrics['recall@6']['mean']:.3f} +/- {metrics['recall@6']['std']:.3f} | "
            f"{metrics['contains']['mean']:.3f} +/- {metrics['contains']['std']:.3f} |"
        )

    lines += [
        "",
        "## 4. Tuned Case Studies",
        "",
        "Top cases where retrieval-tuned SMC-QA improves recall over original SMC-QA:",
        "",
    ]
    for case in cases["improved_vs_original"][:3]:
        lines.append(
            f"- `{case['question_type']}` {case['question']} "
            f"(Original recall {case['original_recall']:.2f} -> Tuned recall {case['tuned_recall']:.2f})"
        )

    lines += [
        "",
        "Top cases where retrieval-tuned SMC-QA regresses against original SMC-QA:",
        "",
    ]
    for case in cases["regressed_vs_original"][:3]:
        lines.append(
            f"- `{case['question_type']}` {case['question']} "
            f"(Original recall {case['original_recall']:.2f} -> Tuned recall {case['tuned_recall']:.2f})"
        )

    with open(RESULTS_DIR / "supplementary_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    raw_data = load_raw_data()
    _, test_data = sample_data(raw_data, seed=42)
    preprocessed = preprocess(test_data, "SUPPLEMENTARY TEST seed 42")

    qtype = question_type_breakdown(preprocessed)
    with open(RESULTS_DIR / "question_type_breakdown_seed42.json", "w") as f:
        json.dump(qtype, f, indent=2, ensure_ascii=False)

    ltm = ltm_quality(preprocessed)
    with open(RESULTS_DIR / "ltm_quality_seed42.json", "w") as f:
        json.dump(ltm, f, indent=2, ensure_ascii=False)

    cases = tuned_case_studies(qtype)
    with open(RESULTS_DIR / "tuned_case_studies_seed42.json", "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    depth = retrieval_depth_ablation_summary()
    with open(RESULTS_DIR / "retrieval_depth_ablation_summary.json", "w") as f:
        json.dump(depth, f, indent=2, ensure_ascii=False)

    write_markdown_summary(qtype, ltm, cases, depth)
    print(f"\nSupplementary analyses saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
