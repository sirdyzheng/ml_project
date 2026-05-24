#!/usr/bin/env python3
"""Tune SMC-QA retrieval top-k settings without editing src/.

The original router retrieves 4 items from STM/LTM in some routes while the
reported metric is Hit@6/Recall@6. This script tests whether widening those
route-specific retrieval depths improves SMC-QA.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENT_DIR = PROJECT_ROOT / "Improvement"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IMPROVEMENT_DIR))

import src.router as router
from src.config import EXPERIMENT_SEEDS
from src.data_loader import load_raw_data, sample_data
from tune_smc import evaluate_config, preprocess

RESULTS_DIR = PROJECT_ROOT / "Improvement" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def evaluate_with_router_topk(preprocessed, config):
    with temporary_router_globals(
        STM_TOP_K=config["stm_top_k"],
        LTM_TOP_K=config["ltm_top_k"],
    ):
        return evaluate_config(preprocessed, config)


def config_grid() -> Iterable[Dict[str, Any]]:
    base_configs = [
        {
            "base": "baseline_thr0.40_top3_auto",
            "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2},
            "threshold": 0.40,
            "promotion_top_k": 3,
            "route": "auto",
        },
        {
            "base": "baseline_thr0.25_top3_auto",
            "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2},
            "threshold": 0.25,
            "promotion_top_k": 3,
            "route": "auto",
        },
        {
            "base": "div_heavy_thr0.40_top5_auto",
            "weights": {"relevance": 0.4, "reuse": 0.3, "diversity": 0.3},
            "threshold": 0.40,
            "promotion_top_k": 5,
            "route": "auto",
        },
        {
            "base": "baseline_thr0.40_top3_hybrid",
            "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2},
            "threshold": 0.40,
            "promotion_top_k": 3,
            "route": "hybrid",
        },
    ]
    topk_pairs = [(4, 4), (6, 4), (4, 6), (6, 6), (8, 8)]
    for base in base_configs:
        for stm_top_k, ltm_top_k in topk_pairs:
            config = dict(base)
            config["stm_top_k"] = stm_top_k
            config["ltm_top_k"] = ltm_top_k
            config["name"] = f"{base['base']}_stm{stm_top_k}_ltm{ltm_top_k}"
            yield config


def tune_on_dev(dev_preprocessed):
    rows = []
    configs = list(config_grid())
    print(f"\n[RETRIEVAL TUNING] evaluating {len(configs)} configs on dev")
    for config in tqdm(configs, desc="retrieval configs"):
        summary, _ = evaluate_with_router_topk(dev_preprocessed, config)
        rows.append({"config": config, "summary": summary})

    rows.sort(
        key=lambda row: (
            row["summary"]["objective"],
            row["summary"]["hit@6"],
            row["summary"]["recall@6"],
            row["summary"]["contains"],
        ),
        reverse=True,
    )
    return rows


def evaluate_test(raw_data, best_configs):
    test_runs = {}
    for seed in EXPERIMENT_SEEDS:
        _, test_data = sample_data(raw_data, seed=seed)
        preprocessed = preprocess(test_data, f"RETRIEVAL TEST seed {seed}")
        seed_results = {}
        for rank, config in enumerate(best_configs, start=1):
            print(f"\n[RETRIEVAL TEST seed {seed}] rank {rank}: {config['name']}")
            summary, metrics = evaluate_with_router_topk(preprocessed, config)
            seed_results[config["name"]] = {"summary": summary, "metrics": metrics}
            print(
                f"Hit@6={summary['hit@6']:.3f} "
                f"Recall@6={summary['recall@6']:.3f} "
                f"Contains={summary['contains']:.3f}"
            )
        test_runs[str(seed)] = seed_results
    return test_runs


def merge(test_runs):
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
    dev_preprocessed = preprocess(dev_data, "RETRIEVAL DEV")

    dev_rows = tune_on_dev(dev_preprocessed)
    with open(RESULTS_DIR / "smc_retrieval_tuning_dev_results.json", "w") as f:
        json.dump(dev_rows, f, indent=2, ensure_ascii=False)

    print("\nTop retrieval-tuned dev configs:")
    for rank, row in enumerate(dev_rows[:10], start=1):
        s = row["summary"]
        print(
            f"{rank:>2}. {row['config']['name']}: "
            f"Hit@6={s['hit@6']:.3f} Recall@6={s['recall@6']:.3f} "
            f"Contains={s['contains']:.3f} Objective={s['objective']:.3f}"
        )

    best_configs = [row["config"] for row in dev_rows[:3]]
    test_runs = evaluate_test(raw_data, best_configs)
    merged = merge(test_runs)
    with open(RESULTS_DIR / "smc_retrieval_tuned_test_results.json", "w") as f:
        json.dump({"test_runs": test_runs, "merged": merged}, f, indent=2, ensure_ascii=False)

    print("\nMerged retrieval-tuned test summaries:")
    for method_name, metrics in merged.items():
        print(
            f"{method_name}: "
            f"Hit@6={metrics['hit@6']['mean']:.3f}+/-{metrics['hit@6']['std']:.3f} "
            f"Recall@6={metrics['recall@6']['mean']:.3f}+/-{metrics['recall@6']['std']:.3f} "
            f"Contains={metrics['contains']['mean']:.3f}+/-{metrics['contains']['std']:.3f}"
        )


if __name__ == "__main__":
    main()
