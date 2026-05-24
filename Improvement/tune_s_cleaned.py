#!/usr/bin/env python3
"""Tune SMC-QA on s_cleaned without editing the original repository code."""

from __future__ import annotations

import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline
import src.router as router
from src.chunking import chunk_turns
from src.data_loader import extract_sessions, get_gold_turns, load_raw_data
from src.embedder import encode_texts
from src.evaluate import contains_match, hit_at_k, recall_at_k

RESULTS_DIR = PROJECT_ROOT / "Improvement" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = 200
SAMPLE_SIZE = 100
TUNING_SIZE = 30

WEIGHT_SPECS = [
    {"name": "baseline", "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2}},
    {"name": "rel_heavy", "weights": {"relevance": 0.5, "reuse": 0.3, "diversity": 0.2}},
    {"name": "div_heavy", "weights": {"relevance": 0.4, "reuse": 0.3, "diversity": 0.3}},
]
THRESHOLDS = [0.20, 0.30, 0.40]
PROMOTION_TOP_KS = [3, 8]
TOPK_PAIRS = [(4, 4), (8, 8)]


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


def sample_s_cleaned():
    raw_data = load_raw_data("longmemeval_s_cleaned.json")
    rng = random.Random(42)
    indices = list(range(len(raw_data)))
    rng.shuffle(indices)
    return [raw_data[i] for i in indices[:SAMPLE_SIZE]]


def preprocess(data_items: List[Dict[str, Any]]):
    print(f"\n[s_cleaned] preprocessing {len(data_items)} items, max_turns={MAX_TURNS}")
    all_texts: List[str] = []
    ranges = []
    for item in tqdm(data_items, desc="s_cleaned chunking"):
        turns = extract_sessions(item, max_turns=MAX_TURNS)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([chunk["text"] for chunk in chunks])
        all_texts.append(item["question"])
        ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"[s_cleaned] encoding {len(all_texts)} texts")
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


def config_grid() -> Iterable[Dict[str, Any]]:
    for weight_spec in WEIGHT_SPECS:
        for threshold in THRESHOLDS:
            for promotion_top_k in PROMOTION_TOP_KS:
                for stm_top_k, ltm_top_k in TOPK_PAIRS:
                    yield {
                        "name": (
                            f"{weight_spec['name']}"
                            f"_thr{threshold:.2f}"
                            f"_top{promotion_top_k}"
                            f"_stm{stm_top_k}_ltm{ltm_top_k}"
                        ),
                        "weights": weight_spec["weights"],
                        "threshold": threshold,
                        "promotion_top_k": promotion_top_k,
                        "stm_top_k": stm_top_k,
                        "ltm_top_k": ltm_top_k,
                        "route": "auto",
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
    with temporary_router_globals(STM_TOP_K=config["stm_top_k"], LTM_TOP_K=config["ltm_top_k"]):
        return router.retrieve_with_routing(
            sample["question"],
            sample["query_emb"],
            stm,
            ltm,
            route=config["route"],
        )


def run_baseline(sample: Dict[str, Any], name: str):
    if name == "Flat Memory":
        return pipeline.run_flat(sample["chunks"], sample["query_emb"])
    if name == "Freq Promotion":
        return pipeline.run_freq_promotion(
            sample["chunks"],
            sample["query_emb"],
            query_text=sample["question"],
        )
    if name == "Original SMC-QA":
        return pipeline.run_smc_qa(
            sample["chunks"],
            sample["query_emb"],
            query_text=sample["question"],
        )
    raise ValueError(name)


def eval_retrieved(retrieved, sample: Dict[str, Any]):
    pred_text = " ".join(item.text for item in retrieved[:3])
    return {
        "hit@6": hit_at_k(retrieved, sample["gold_texts"]),
        "recall@6": recall_at_k(retrieved, sample["gold_texts"]),
        "contains": contains_match(pred_text, sample["answer"]),
    }


def evaluate_config(preprocessed: List[Dict[str, Any]], config: Dict[str, Any]):
    metrics = {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
    for sample in preprocessed:
        t0 = time.time()
        retrieved = run_tuned_smc(sample, config)
        metrics["latency"].append(time.time() - t0)
        result = eval_retrieved(retrieved, sample)
        for key, value in result.items():
            metrics[key].append(value)
    summary = {key: float(np.mean(values)) for key, values in metrics.items()}
    summary["objective"] = summary["hit@6"] + 0.6 * summary["recall@6"] + 0.3 * summary["contains"]
    return summary, metrics


def evaluate_baselines(preprocessed: List[Dict[str, Any]]):
    output = {}
    for name in ["Flat Memory", "Freq Promotion", "Original SMC-QA"]:
        metrics = {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
        for sample in tqdm(preprocessed, desc=name):
            t0 = time.time()
            retrieved = run_baseline(sample, name)
            metrics["latency"].append(time.time() - t0)
            result = eval_retrieved(retrieved, sample)
            for key, value in result.items():
                metrics[key].append(value)
        output[name] = {
            "summary": {key: float(np.mean(values)) for key, values in metrics.items()},
            "metrics": metrics,
        }
    return output


def main():
    data_items = sample_s_cleaned()
    preprocessed = preprocess(data_items)
    tuning = preprocessed[:TUNING_SIZE]
    validation = preprocessed[TUNING_SIZE:]

    configs = list(config_grid())
    print(f"\n[s_cleaned tuning] {len(configs)} configs on first {len(tuning)} items")
    tuning_rows = []
    for config in tqdm(configs, desc="s_cleaned configs"):
        summary, _ = evaluate_config(tuning, config)
        tuning_rows.append({"config": config, "summary": summary})
    tuning_rows.sort(
        key=lambda row: (
            row["summary"]["objective"],
            row["summary"]["hit@6"],
            row["summary"]["recall@6"],
            row["summary"]["contains"],
        ),
        reverse=True,
    )

    top_configs = [row["config"] for row in tuning_rows[:5]]
    print("\nTop s_cleaned tuning configs:")
    for rank, row in enumerate(tuning_rows[:10], start=1):
        s = row["summary"]
        print(
            f"{rank:>2}. {row['config']['name']}: "
            f"Hit@6={s['hit@6']:.3f} Recall@6={s['recall@6']:.3f} "
            f"Contains={s['contains']:.3f} Objective={s['objective']:.3f}"
        )

    validation_results = {}
    for config in top_configs:
        summary, metrics = evaluate_config(validation, config)
        validation_results[config["name"]] = {
            "config": config,
            "summary": summary,
            "metrics": metrics,
        }
        print(
            f"[validation] {config['name']}: "
            f"Hit@6={summary['hit@6']:.3f} Recall@6={summary['recall@6']:.3f} "
            f"Contains={summary['contains']:.3f}"
        )

    best_validation_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["summary"]["objective"],
            validation_results[name]["summary"]["hit@6"],
            validation_results[name]["summary"]["recall@6"],
        ),
    )
    best_config = validation_results[best_validation_name]["config"]
    best_summary, best_metrics = evaluate_config(preprocessed, best_config)
    baselines = evaluate_baselines(preprocessed)

    output = {
        "settings": {
            "sample_size": SAMPLE_SIZE,
            "tuning_size": TUNING_SIZE,
            "validation_size": SAMPLE_SIZE - TUNING_SIZE,
            "max_turns": MAX_TURNS,
        },
        "tuning_results": tuning_rows,
        "validation_results": validation_results,
        "best_config": best_config,
        "full_100": {
            "Tuned SMC-QA": {"summary": best_summary, "metrics": best_metrics},
            **baselines,
        },
    }
    with open(RESULTS_DIR / "s_cleaned_tuning_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nFull 100-item s_cleaned comparison:")
    for name, payload in output["full_100"].items():
        s = payload["summary"]
        print(
            f"{name}: Hit@6={s['hit@6']:.3f} "
            f"Recall@6={s['recall@6']:.3f} Contains={s['contains']:.3f}"
        )


if __name__ == "__main__":
    main()
