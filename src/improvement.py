"""Improvement experiment utilities for SMC-QA.

This module promotes the supplementary experiments into reusable project code.
It keeps the base pipeline unchanged and exposes tuned SMC-QA variants through
explicit configuration dictionaries.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from . import pipeline
from .chunking import chunk_turns
from .config import (
    CONSOLIDATION_FREQ,
    EXPERIMENT_SEEDS,
    LTM_MAX_SIZE,
    RESULTS_DIR,
    RETRIEVAL_TOP_K,
    SCORE_THRESHOLD,
    SCORE_WEIGHTS,
    STM_CAPACITY,
)
from .data_loader import extract_sessions, get_gold_turns, load_raw_data, sample_data
from .embedder import cosine_sim, encode_texts
from .evaluate import contains_match, exact_match, hit_at_k, normalize_text, recall_at_k
from .memory import FlatMemory, LTM, STM, MemoryItem
from .promotion import frequency_promote, promote
from .router import route_query


BASE_SMC_CONFIG = {
    "name": "Base SMC-QA",
    "weights": SCORE_WEIGHTS,
    "threshold": SCORE_THRESHOLD,
    "promotion_top_k": 3,
    "route": "auto",
    "stm_top_k": 4,
    "ltm_top_k": 4,
    "final_top_k": RETRIEVAL_TOP_K,
}

RETRIEVAL_TUNED_CONFIG = {
    "name": "Retrieval-tuned SMC-QA",
    "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2},
    "threshold": 0.40,
    "promotion_top_k": 3,
    "route": "auto",
    "stm_top_k": 8,
    "ltm_top_k": 8,
    "final_top_k": RETRIEVAL_TOP_K,
}

S_CLEANED_TUNED_CONFIG = {
    "name": "Tuned SMC-QA",
    "weights": {"relevance": 0.5, "reuse": 0.3, "diversity": 0.2},
    "threshold": 0.20,
    "promotion_top_k": 3,
    "route": "auto",
    "stm_top_k": 8,
    "ltm_top_k": 8,
    "final_top_k": RETRIEVAL_TOP_K,
}


def save_json(payload: Any, filename: str, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(to_serializable(payload), f, indent=2, ensure_ascii=False)
    return path


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def preprocess_items(
    data_items: List[Dict[str, Any]],
    label: str,
    max_turns: int = 0,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Chunk and embed a list of LongMemEval samples once for many methods."""
    print(f"\n[{label}] preprocessing {len(data_items)} items")
    all_texts: List[str] = []
    ranges: List[Tuple[int, int, List[Dict[str, Any]], Dict[str, Any]]] = []

    iterator = tqdm(data_items, desc=f"{label} chunking") if show_progress else data_items
    for item in iterator:
        turns = extract_sessions(item, max_turns=max_turns)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend(chunk["text"] for chunk in chunks)
        all_texts.append(item["question"])
        ranges.append((start, len(all_texts) - 1, chunks, item))

    print(f"[{label}] encoding {len(all_texts)} texts")
    all_embs = encode_texts(all_texts, batch_size=128, show_progress=show_progress)

    preprocessed = []
    for start, query_idx, chunks, item in ranges:
        for idx, chunk in enumerate(chunks):
            chunk["embedding"] = all_embs[start + idx]
        preprocessed.append(
            {
                "question_id": item.get("question_id", ""),
                "question_type": item.get("question_type", ""),
                "question": item["question"],
                "answer": str(item["answer"]),
                "chunks": chunks,
                "query_emb": all_embs[query_idx],
                "gold_texts": get_gold_turns(item),
            }
        )
    return preprocessed


def build_memory(
    chunks: List[Dict[str, Any]],
    query_emb: np.ndarray,
    promotion_fn: str = "smc",
    threshold: float = SCORE_THRESHOLD,
    weights: Dict[str, float] = SCORE_WEIGHTS,
    promotion_top_k: int = 3,
) -> Tuple[STM, LTM]:
    """Build STM/LTM without mutating global pipeline settings."""
    stm = STM(STM_CAPACITY)
    ltm = LTM(LTM_MAX_SIZE)
    recent_embs = [query_emb]

    for idx, chunk in enumerate(chunks):
        item = MemoryItem(
            chunk["chunk_id"],
            chunk["text"],
            chunk["embedding"],
            chunk["session_id"],
            chunk["has_answer"],
        )
        stm.add(item)
        recent_embs.append(chunk["embedding"])
        if len(recent_embs) > 5:
            recent_embs.pop(0)

        if (idx + 1) % 3 == 0 and stm.items:
            stm.retrieve(query_emb, 2)

        if (idx + 1) % CONSOLIDATION_FREQ == 0:
            if promotion_fn == "smc":
                promote(stm, ltm, recent_embs, top_k=promotion_top_k, threshold=threshold, weights=weights)
            elif promotion_fn == "freq":
                frequency_promote(stm, ltm, promotion_top_k)
            else:
                raise ValueError(f"Unknown promotion_fn: {promotion_fn}")

    if promotion_fn == "smc":
        promote(stm, ltm, recent_embs, top_k=promotion_top_k, threshold=threshold, weights=weights)
    elif promotion_fn == "freq":
        frequency_promote(stm, ltm, promotion_top_k)
    return stm, ltm


def retrieve_with_config(
    query_text: str,
    query_emb: np.ndarray,
    stm: STM,
    ltm: LTM,
    route: str = "auto",
    stm_top_k: int = 4,
    ltm_top_k: int = 4,
    final_top_k: int = RETRIEVAL_TOP_K,
) -> List[MemoryItem]:
    if route == "auto":
        route = route_query(query_text, query_emb, stm, ltm)

    if route == "stm":
        return stm.retrieve(query_emb, stm_top_k)
    if route == "ltm":
        return ltm.retrieve(query_emb, ltm_top_k)

    stm_results = stm.retrieve(query_emb, stm_top_k)
    ltm_results = ltm.retrieve(query_emb, ltm_top_k)
    seen_ids = set()
    merged = []
    for item in stm_results + ltm_results:
        if item.chunk_id not in seen_ids:
            seen_ids.add(item.chunk_id)
            merged.append(item)
    if len(merged) <= final_top_k:
        return merged

    embs = np.stack([item.embedding for item in merged])
    sims = cosine_sim(query_emb, embs).flatten()
    top_indices = np.argsort(sims)[::-1][:final_top_k]
    return [merged[i] for i in top_indices]


def run_tuned_smc(sample: Dict[str, Any], config: Dict[str, Any]) -> List[MemoryItem]:
    stm, ltm = build_memory(
        sample["chunks"],
        sample["query_emb"],
        promotion_fn="smc",
        threshold=config["threshold"],
        weights=config["weights"],
        promotion_top_k=config["promotion_top_k"],
    )
    return retrieve_with_config(
        sample["question"],
        sample["query_emb"],
        stm,
        ltm,
        route=config.get("route", "auto"),
        stm_top_k=config.get("stm_top_k", 4),
        ltm_top_k=config.get("ltm_top_k", 4),
        final_top_k=config.get("final_top_k", RETRIEVAL_TOP_K),
    )


def run_flat_memory(sample: Dict[str, Any]) -> List[MemoryItem]:
    memory = FlatMemory()
    for chunk in sample["chunks"]:
        memory.add(
            MemoryItem(
                chunk["chunk_id"],
                chunk["text"],
                chunk["embedding"],
                chunk["session_id"],
                chunk["has_answer"],
            )
        )
    return memory.retrieve(sample["query_emb"], RETRIEVAL_TOP_K)


def run_method(sample: Dict[str, Any], method_name: str) -> List[MemoryItem]:
    if method_name == "Flat Memory":
        return run_flat_memory(sample)
    if method_name == "Freq Promotion":
        stm, ltm = build_memory(sample["chunks"], sample["query_emb"], promotion_fn="freq")
        return retrieve_with_config(sample["question"], sample["query_emb"], stm, ltm, route="hybrid")
    if method_name == "Base SMC-QA":
        return run_tuned_smc(sample, BASE_SMC_CONFIG)
    if method_name == "Retrieval-tuned SMC-QA":
        return run_tuned_smc(sample, RETRIEVAL_TUNED_CONFIG)
    if method_name == "Tuned SMC-QA":
        return run_tuned_smc(sample, S_CLEANED_TUNED_CONFIG)
    raise ValueError(f"Unknown method: {method_name}")


def eval_retrieved(retrieved: List[MemoryItem], sample: Dict[str, Any]) -> Dict[str, float]:
    pred_text = " ".join(item.text for item in retrieved[:3])
    return {
        "hit@6": hit_at_k(retrieved, sample["gold_texts"]),
        "recall@6": recall_at_k(retrieved, sample["gold_texts"]),
        "contains": contains_match(pred_text, sample["answer"]),
    }


def summarize_metric_lists(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: mean(values) for key, values in metrics.items()}


def evaluate_config(preprocessed: List[Dict[str, Any]], config: Dict[str, Any]):
    metrics = {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
    for sample in preprocessed:
        t0 = time.time()
        retrieved = run_tuned_smc(sample, config)
        metrics["latency"].append(time.time() - t0)
        result = eval_retrieved(retrieved, sample)
        for metric, value in result.items():
            metrics[metric].append(value)

    summary = summarize_metric_lists(metrics)
    summary["objective"] = summary["hit@6"] + 0.6 * summary["recall@6"] + 0.3 * summary["contains"]
    return summary, metrics


def evaluate_methods(preprocessed: List[Dict[str, Any]], methods: Sequence[str]):
    output = {}
    for method in methods:
        metrics = {"hit@6": [], "recall@6": [], "contains": [], "latency": []}
        for sample in tqdm(preprocessed, desc=method):
            t0 = time.time()
            retrieved = run_method(sample, method)
            metrics["latency"].append(time.time() - t0)
            result = eval_retrieved(retrieved, sample)
            for key, value in result.items():
                metrics[key].append(value)
        output[method] = {"summary": summarize_metric_lists(metrics), "metrics": metrics}
    return output


def promotion_config_grid() -> Iterable[Dict[str, Any]]:
    weight_grid = [
        {"name": "baseline", "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2}},
        {"name": "rel_heavy", "weights": {"relevance": 0.5, "reuse": 0.3, "diversity": 0.2}},
        {"name": "rel_strong", "weights": {"relevance": 0.6, "reuse": 0.2, "diversity": 0.2}},
        {"name": "div_heavy", "weights": {"relevance": 0.4, "reuse": 0.3, "diversity": 0.3}},
    ]
    for weight_spec in weight_grid:
        for threshold in [0.25, 0.30, 0.35, 0.40]:
            for promotion_top_k in [3, 5]:
                for route in ["auto", "hybrid"]:
                    yield {
                        "name": f"{weight_spec['name']}_thr{threshold:.2f}_top{promotion_top_k}_{route}",
                        "weights": weight_spec["weights"],
                        "threshold": threshold,
                        "promotion_top_k": promotion_top_k,
                        "route": route,
                        "stm_top_k": 4,
                        "ltm_top_k": 4,
                        "final_top_k": RETRIEVAL_TOP_K,
                    }


def retrieval_config_grid() -> Iterable[Dict[str, Any]]:
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
    for base in base_configs:
        for stm_top_k, ltm_top_k in [(4, 4), (6, 4), (4, 6), (6, 6), (8, 8)]:
            config = dict(base)
            config.update(
                {
                    "stm_top_k": stm_top_k,
                    "ltm_top_k": ltm_top_k,
                    "final_top_k": RETRIEVAL_TOP_K,
                    "name": f"{base['base']}_stm{stm_top_k}_ltm{ltm_top_k}",
                }
            )
            yield config


def s_cleaned_config_grid() -> Iterable[Dict[str, Any]]:
    weight_specs = [
        {"name": "baseline", "weights": {"relevance": 0.4, "reuse": 0.4, "diversity": 0.2}},
        {"name": "rel_heavy", "weights": {"relevance": 0.5, "reuse": 0.3, "diversity": 0.2}},
        {"name": "div_heavy", "weights": {"relevance": 0.4, "reuse": 0.3, "diversity": 0.3}},
    ]
    for weight_spec in weight_specs:
        for threshold in [0.20, 0.30, 0.40]:
            for promotion_top_k in [3, 8]:
                for stm_top_k, ltm_top_k in [(4, 4), (8, 8)]:
                    yield {
                        "name": (
                            f"{weight_spec['name']}_thr{threshold:.2f}"
                            f"_top{promotion_top_k}_stm{stm_top_k}_ltm{ltm_top_k}"
                        ),
                        "weights": weight_spec["weights"],
                        "threshold": threshold,
                        "promotion_top_k": promotion_top_k,
                        "stm_top_k": stm_top_k,
                        "ltm_top_k": ltm_top_k,
                        "route": "auto",
                        "final_top_k": RETRIEVAL_TOP_K,
                    }


def tune_configs(preprocessed: List[Dict[str, Any]], configs: Iterable[Dict[str, Any]], label: str):
    rows = []
    config_list = list(configs)
    print(f"\n[{label}] evaluating {len(config_list)} configs")
    for config in tqdm(config_list, desc=label):
        summary, _ = evaluate_config(preprocessed, config)
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


def evaluate_test_seeds(raw_data: List[Dict[str, Any]], configs: Sequence[Dict[str, Any]], label: str):
    test_runs = {}
    for seed in EXPERIMENT_SEEDS:
        _, test_data = sample_data(raw_data, seed=seed)
        preprocessed = preprocess_items(test_data, f"{label} seed {seed}")
        seed_results = {}
        for rank, config in enumerate(configs, start=1):
            print(f"\n[{label} seed {seed}] rank {rank}: {config['name']}")
            summary, metrics = evaluate_config(preprocessed, config)
            seed_results[config["name"]] = {"summary": summary, "metrics": metrics}
            print(
                f"Hit@6={summary['hit@6']:.3f} "
                f"Recall@6={summary['recall@6']:.3f} "
                f"Contains={summary['contains']:.3f}"
            )
        test_runs[str(seed)] = seed_results
    return test_runs


def merge_seed_summaries(test_runs: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    method_names = next(iter(test_runs.values())).keys()
    for method_name in method_names:
        merged[method_name] = {}
        for metric in ["hit@6", "recall@6", "contains", "latency", "objective"]:
            values = [test_runs[str(seed)][method_name]["summary"][metric] for seed in EXPERIMENT_SEEDS]
            merged[method_name][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return merged


def sample_s_cleaned(sample_size: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    raw_data = load_raw_data("longmemeval_s_cleaned.json")
    rng = random.Random(seed)
    indices = list(range(len(raw_data)))
    rng.shuffle(indices)
    return [raw_data[idx] for idx in indices[:sample_size]]


def question_type_breakdown(preprocessed: List[Dict[str, Any]]):
    methods = ["Flat Memory", "Freq Promotion", "Base SMC-QA", "Retrieval-tuned SMC-QA"]
    by_type = {
        method: defaultdict(lambda: {"hit@6": [], "recall@6": [], "contains": [], "count": 0})
        for method in methods
    }
    per_sample = []

    print("\n[question type] evaluating methods")
    for sample in tqdm(preprocessed, desc="question type"):
        row = {
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
            row["methods"][method] = {
                **metrics,
                "top3": [item.text[:240] for item in retrieved[:3]],
            }
            bucket = by_type[method][sample["question_type"]]
            bucket["count"] += 1
            for metric in ["hit@6", "recall@6", "contains"]:
                bucket[metric].append(metrics[metric])
        per_sample.append(row)

    summarized = {}
    for method, type_map in by_type.items():
        summarized[method] = {}
        for question_type, metrics in sorted(type_map.items()):
            summarized[method][question_type] = {
                "count": metrics["count"],
                **summarize_metric_lists({k: v for k, v in metrics.items() if isinstance(v, list)}),
            }
    return {"by_type": summarized, "per_sample": per_sample}


def ltm_for_variant(sample: Dict[str, Any], variant: str) -> LTM:
    if variant == "Freq Promotion":
        return build_memory(sample["chunks"], sample["query_emb"], "freq")[1]
    if variant == "Base SMC-QA":
        return build_memory(
            sample["chunks"],
            sample["query_emb"],
            "smc",
            threshold=BASE_SMC_CONFIG["threshold"],
            weights=BASE_SMC_CONFIG["weights"],
            promotion_top_k=BASE_SMC_CONFIG["promotion_top_k"],
        )[1]
    if variant == "Retrieval-tuned SMC-QA":
        return build_memory(
            sample["chunks"],
            sample["query_emb"],
            "smc",
            threshold=RETRIEVAL_TUNED_CONFIG["threshold"],
            weights=RETRIEVAL_TUNED_CONFIG["weights"],
            promotion_top_k=RETRIEVAL_TUNED_CONFIG["promotion_top_k"],
        )[1]
    raise ValueError(f"Unknown LTM variant: {variant}")


def ltm_quality(preprocessed: List[Dict[str, Any]]):
    variants = ["Freq Promotion", "Base SMC-QA", "Retrieval-tuned SMC-QA"]
    rows = []
    print("\n[LTM quality] evaluating memory quality")
    for sample in tqdm(preprocessed, desc="ltm quality"):
        for variant in variants:
            ltm = ltm_for_variant(sample, variant)
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
        for question_type in sorted({row["question_type"] for row in variant_rows}):
            subset = [row for row in variant_rows if row["question_type"] == question_type]
            by_type[variant][question_type] = {
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


def tuned_case_studies(question_type_results: Dict[str, Any]):
    cases = []
    for row in question_type_results["per_sample"]:
        base = row["methods"]["Base SMC-QA"]
        tuned = row["methods"]["Retrieval-tuned SMC-QA"]
        freq = row["methods"]["Freq Promotion"]
        cases.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
                "base_hit": base["hit@6"],
                "tuned_hit": tuned["hit@6"],
                "freq_hit": freq["hit@6"],
                "base_recall": base["recall@6"],
                "tuned_recall": tuned["recall@6"],
                "freq_recall": freq["recall@6"],
                "delta_recall_tuned_vs_base": tuned["recall@6"] - base["recall@6"],
                "delta_recall_tuned_vs_freq": tuned["recall@6"] - freq["recall@6"],
                "base_top3": base["top3"],
                "tuned_top3": tuned["top3"],
                "freq_top3": freq["top3"],
            }
        )
    improved = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_base"] > 0],
        key=lambda case: (case["delta_recall_tuned_vs_base"], case["tuned_hit"]),
        reverse=True,
    )
    regressed = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_base"] < 0],
        key=lambda case: case["delta_recall_tuned_vs_base"],
    )
    freq_better = sorted(
        [case for case in cases if case["delta_recall_tuned_vs_freq"] < 0],
        key=lambda case: case["delta_recall_tuned_vs_freq"],
    )
    return {
        "improved_vs_base": improved[:10],
        "regressed_vs_base": regressed[:10],
        "freq_better_than_tuned": freq_better[:10],
    }


def retrieval_depth_ablation_summary(results_dir: Path = RESULTS_DIR):
    dev_path = results_dir / "smc_retrieval_tuning_dev_results.json"
    test_path = results_dir / "smc_retrieval_tuned_test_results.json"
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


def write_markdown_summary(qtype, ltm, cases, depth, results_dir: Path = RESULTS_DIR):
    lines = [
        "# Supplementary Experiment Summary",
        "",
        "## 1. Question Type Breakdown",
        "",
        "Seed 42 test split; methods: Flat Memory, Freq Promotion, Base SMC-QA, Retrieval-tuned SMC-QA.",
        "",
    ]
    for method in ["Base SMC-QA", "Retrieval-tuned SMC-QA", "Freq Promotion"]:
        lines += [
            f"### {method}",
            "",
            "| Question Type | Count | Hit@6 | Recall@6 | Contains |",
            "|---------------|------:|------:|----------:|---------:|",
        ]
        for question_type, metrics in qtype["by_type"][method].items():
            lines.append(
                f"| {question_type} | {metrics['count']} | "
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
        "| Config | STM Top-K | LTM Top-K | Hit@6 | Recall@6 | Contains |",
        "|--------|----------:|----------:|------:|----------:|---------:|",
    ]
    for row in depth["dev_top10"][:5]:
        lines.append(
            f"| {row['name']} | {row['stm_top_k']} | {row['ltm_top_k']} | "
            f"{row['hit@6']:.3f} | {row['recall@6']:.3f} | {row['contains']:.3f} |"
        )

    lines += ["", "Merged test results for top retrieval-depth configs:", ""]
    lines += ["| Config | Hit@6 | Recall@6 | Contains |", "|--------|------:|----------:|---------:|"]
    for name, metrics in depth["test_merged"].items():
        lines.append(
            f"| {name} | {metrics['hit@6']['mean']:.3f} +/- {metrics['hit@6']['std']:.3f} | "
            f"{metrics['recall@6']['mean']:.3f} +/- {metrics['recall@6']['std']:.3f} | "
            f"{metrics['contains']['mean']:.3f} +/- {metrics['contains']['std']:.3f} |"
        )

    lines += ["", "## 4. Tuned Case Studies", ""]
    lines += ["Top cases where retrieval-tuned SMC-QA improves recall over Base SMC-QA:", ""]
    for case in cases["improved_vs_base"][:3]:
        lines.append(
            f"- `{case['question_type']}` {case['question']} "
            f"(Base recall {case['base_recall']:.2f} -> Tuned recall {case['tuned_recall']:.2f})"
        )
    lines += ["", "Top cases where retrieval-tuned SMC-QA regresses against Base SMC-QA:", ""]
    for case in cases["regressed_vs_base"][:3]:
        lines.append(
            f"- `{case['question_type']}` {case['question']} "
            f"(Base recall {case['base_recall']:.2f} -> Tuned recall {case['tuned_recall']:.2f})"
        )

    return save_text("\n".join(lines) + "\n", "supplementary_summary.md", results_dir)


def save_text(content: str, filename: str, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    with open(path, "w") as f:
        f.write(content)
    return path


def build_prompt(sample: Dict[str, Any], retrieved: Sequence[MemoryItem]) -> str:
    context = "\n\n".join(f"[{idx}] {item.text}" for idx, item in enumerate(retrieved[:6], start=1))
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, answer with 'I don't know'. "
        "Keep the answer concise.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return str(text).strip()
    try:
        chunks = []
        for item in response.output:
            for content in item.content:
                if getattr(content, "type", "") in {"output_text", "text"}:
                    chunks.append(content.text)
        return "\n".join(chunks).strip()
    except Exception:
        return str(response)


def token_overlap_score(predicted: str, gold: str) -> float:
    pred_words = set(normalize_text(predicted).split())
    gold_words = set(normalize_text(gold).split())
    if not gold_words:
        return 0.0
    return len(pred_words & gold_words) / len(gold_words)


def evaluate_answer(generated: str, gold: str) -> Dict[str, float]:
    return {
        "contains": contains_match(generated, gold),
        "exact_match": exact_match(generated, gold),
        "token_overlap": token_overlap_score(generated, gold),
    }


def run_answer_generation(
    preprocessed: List[Dict[str, Any]],
    methods: Sequence[str],
    model: str,
    run_api: bool,
) -> Dict[str, Any]:
    can_call_api = run_api and bool(os.environ.get("OPENAI_API_KEY"))
    client = None
    if can_call_api:
        try:
            from openai import OpenAI

            client = OpenAI()
        except Exception as exc:
            print(f"OpenAI client unavailable; falling back to dry run: {exc}")
            can_call_api = False
    elif run_api:
        print("OPENAI_API_KEY is not set; falling back to dry run.")

    results = {
        "settings": {
            "n": len(preprocessed),
            "methods": list(methods),
            "model": model,
            "run_api": bool(can_call_api),
        },
        "items": [],
    }

    for sample in tqdm(preprocessed, desc="answer generation"):
        row = {
            "question_id": sample["question_id"],
            "question_type": sample["question_type"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "methods": {},
        }
        for method in methods:
            retrieved = run_method(sample, method)
            prompt = build_prompt(sample, retrieved)
            retrieval_metrics = {
                "hit@6": hit_at_k(retrieved, sample["gold_texts"]),
                "recall@6": recall_at_k(retrieved, sample["gold_texts"]),
            }
            generated = ""
            answer_metrics = {}
            error = None
            if can_call_api and client is not None:
                try:
                    t0 = time.time()
                    response = client.responses.create(model=model, input=prompt, temperature=0)
                    generated = response_text(response)
                    answer_metrics = evaluate_answer(generated, sample["answer"])
                    answer_metrics["latency"] = time.time() - t0
                except Exception as exc:
                    error = str(exc)
            row["methods"][method] = {
                "retrieval": retrieval_metrics,
                "prompt": prompt,
                "generated_answer": generated,
                "answer_metrics": answer_metrics,
                "error": error,
            }
        results["items"].append(row)

    if can_call_api:
        summary = {}
        for method in methods:
            method_rows = [item["methods"][method] for item in results["items"]]
            summary[method] = {}
            for metric in ["contains", "exact_match", "token_overlap", "latency"]:
                values = [row["answer_metrics"][metric] for row in method_rows if metric in row["answer_metrics"]]
                summary[method][metric] = mean(values)
        results["summary"] = summary
    else:
        results["summary"] = {"dry_run": "No API calls were made. Prompts are saved for later execution."}
    return results
