#!/usr/bin/env python3
"""Tune SMC-QA without modifying the original repository code.

This script keeps all outputs under Improvement/results and monkey-patches only
in-process globals when evaluating a candidate configuration.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline
from src.chunking import chunk_turns
from src.config import EXPERIMENT_SEEDS
from src.data_loader import extract_sessions, get_gold_turns, load_raw_data, sample_data
from src.embedder import encode_texts
from src.evaluate import contains_match, hit_at_k, recall_at_k
from src.router import retrieve_with_routing

RESULTS_DIR = PROJECT_ROOT / "Improvement" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


WEIGHT_GRID = [
    {"name": "baseline", "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2}},
    {"name": "rel_heavy", "weights": {"relevance": 0.5, "reuse": 0.3, "diversity": 0.2}},
    {"name": "rel_strong", "weights": {"relevance": 0.6, "reuse": 0.2, "diversity": 0.2}},
    {"name": "div_heavy", "weights": {"relevance": 0.4, "reuse": 0.3, "diversity": 0.3}},
]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40]
PROMOTION_TOP_KS = [3, 5]
ROUTES = ["auto", "hybrid"]


@contextmanager
def temporary_pipeline_globals(**updates):
    old_values = {name: getattr(pipeline, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(pipeline, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(pipeline, name, value)


def preprocess(data_items: List[Dict[str, Any]], label: str):
    print(f"\n[{label}] preprocessing {len(data_items)} items")
    all_texts: List[str] = []
    ranges: List[Tuple[int, int, List[Dict[str, Any]], Dict[str, Any]]] = []

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
                "chunks": chunks,
                "question": item["question"],
                "question_type": item.get("question_type", ""),
                "query_emb": all_embs[query_idx],
                "gold_texts": get_gold_turns(item),
                "answer": str(item["answer"]),
            }
        )
    return preprocessed


def config_grid() -> Iterable[Dict[str, Any]]:
    for weight_spec in WEIGHT_GRID:
        for threshold in THRESHOLDS:
            for promotion_top_k in PROMOTION_TOP_KS:
                for route in ROUTES:
                    yield {
                        "name": (
                            f"{weight_spec['name']}"
                            f"_thr{threshold:.2f}"
                            f"_top{promotion_top_k}"
                            f"_{route}"
                        ),
                        "weights": weight_spec["weights"],
                        "threshold": threshold,
                        "promotion_top_k": promotion_top_k,
                        "route": route,
                    }


def run_tuned_smc(sample: Dict[str, Any], config: Dict[str, Any]):
    with temporary_pipeline_globals(PROMOTION_TOP_K=config["promotion_top_k"]):
        stm, ltm = pipeline._build_memory(
            sample["chunks"],
            sample["query_emb"],
            promotion_fn="smc",
            threshold=config["threshold"],
            weights=config["weights"],
        )
    return retrieve_with_routing(
        sample["question"],
        sample["query_emb"],
        stm,
        ltm,
        route=config["route"],
    )


def evaluate_config(preprocessed: List[Dict[str, Any]], config: Dict[str, Any]):
    metrics = {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
    for sample in preprocessed:
        t0 = time.time()
        retrieved = run_tuned_smc(sample, config)
        metrics["latency"].append(time.time() - t0)
        metrics["hit@6"].append(hit_at_k(retrieved, sample["gold_texts"]))
        metrics["recall@6"].append(recall_at_k(retrieved, sample["gold_texts"]))
        pred_text = " ".join(item.text for item in retrieved[:3])
        metrics["contains"].append(contains_match(pred_text, sample["answer"]))

    summary = {metric: float(np.mean(values)) for metric, values in metrics.items()}
    summary["objective"] = (
        summary["hit@6"] * 1.0
        + summary["recall@6"] * 0.6
        + summary["contains"] * 0.3
    )
    return summary, metrics


def tune_on_dev(dev_preprocessed):
    results = []
    configs = list(config_grid())
    print(f"\n[TUNING] evaluating {len(configs)} SMC-QA configs on dev")
    for config in tqdm(configs, desc="tuning configs"):
        summary, _ = evaluate_config(dev_preprocessed, config)
        results.append({"config": config, "summary": summary})

    results.sort(
        key=lambda row: (
            row["summary"]["objective"],
            row["summary"]["hit@6"],
            row["summary"]["recall@6"],
            row["summary"]["contains"],
        ),
        reverse=True,
    )
    return results


def evaluate_test_seeds(raw_data, best_configs):
    test_runs = {}
    for seed in EXPERIMENT_SEEDS:
        _, test_data = sample_data(raw_data, seed=seed)
        preprocessed = preprocess(test_data, f"TEST seed {seed}")
        seed_results = {}
        for rank, config in enumerate(best_configs, start=1):
            print(f"\n[TEST seed {seed}] rank {rank}: {config['name']}")
            summary, metrics = evaluate_config(preprocessed, config)
            seed_results[config["name"]] = {"summary": summary, "metrics": metrics}
            print(
                f"Hit@6={summary['hit@6']:.3f} "
                f"Recall@6={summary['recall@6']:.3f} "
                f"Contains={summary['contains']:.3f} "
                f"Latency={summary['latency'] * 1000:.1f}ms"
            )
        test_runs[str(seed)] = seed_results
    return test_runs


def merge_seed_summaries(test_runs):
    merged = {}
    method_names = next(iter(test_runs.values())).keys()
    for method_name in method_names:
        merged[method_name] = {}
        for metric in ["hit@6", "recall@6", "contains", "latency", "objective"]:
            values = [
                test_runs[str(seed)][method_name]["summary"][metric]
                for seed in EXPERIMENT_SEEDS
            ]
            merged[method_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return merged


def main():
    raw_data = load_raw_data()
    dev_data, _ = sample_data(raw_data)
    dev_preprocessed = preprocess(dev_data, "DEV")

    tuning_results = tune_on_dev(dev_preprocessed)
    top_configs = [row["config"] for row in tuning_results[:3]]

    with open(RESULTS_DIR / "smc_tuning_dev_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2, ensure_ascii=False)

    print("\nTop dev configs:")
    for rank, row in enumerate(tuning_results[:10], start=1):
        s = row["summary"]
        print(
            f"{rank:>2}. {row['config']['name']}: "
            f"Hit@6={s['hit@6']:.3f} Recall@6={s['recall@6']:.3f} "
            f"Contains={s['contains']:.3f} Objective={s['objective']:.3f}"
        )

    test_runs = evaluate_test_seeds(raw_data, top_configs)
    merged = merge_seed_summaries(test_runs)

    with open(RESULTS_DIR / "smc_tuned_test_results.json", "w") as f:
        json.dump({"test_runs": test_runs, "merged": merged}, f, indent=2, ensure_ascii=False)

    print("\nMerged test summaries:")
    for method_name, metrics in merged.items():
        print(
            f"{method_name}: "
            f"Hit@6={metrics['hit@6']['mean']:.3f}+/-{metrics['hit@6']['std']:.3f} "
            f"Recall@6={metrics['recall@6']['mean']:.3f}+/-{metrics['recall@6']['std']:.3f} "
            f"Contains={metrics['contains']['mean']:.3f}+/-{metrics['contains']['std']:.3f}"
        )


if __name__ == "__main__":
    main()
