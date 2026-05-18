#!/usr/bin/env python3
"""Main entry — run everything: dev tuning, formal experiments, ablation."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data, sample_data
from src.pipeline import (
    run_experiment, summarize_results, save_results,
    METHODS, ABLATION_METHODS,
)
from src.config import EXPERIMENT_SEEDS, RESULTS_DIR


def print_table(all_runs_summary, title="Results"):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Method':<25} {'Hit@6':>10} {'Recall@6':>10} {'Contains':>10} {'Latency(ms)':>12}"
    print(header)
    print("-" * 80)
    for method, metrics in all_runs_summary.items():
        h = metrics["hit@6"]
        r = metrics["recall@6"]
        c = metrics["contains"]
        l = metrics["latency"]
        print(f"{method:<25} {h['mean']:.3f}±{h['std']:.3f} {r['mean']:.3f}±{r['std']:.3f} "
              f"{c['mean']:.3f}±{c['std']:.3f} {l['mean']*1000:.1f}±{l['std']*1000:.1f}")
    print("=" * 80)


def merge_multi_seed_results(results_list):
    """Merge results from multiple seeds into mean±std."""
    merged = {}
    all_methods = results_list[0].keys()
    for method in all_methods:
        merged[method] = {}
        for metric in results_list[0][method]:
            values_per_seed = [np.mean(r[method][metric]) for r in results_list]
            merged[method][metric] = {
                "mean": float(np.mean(values_per_seed)),
                "std": float(np.std(values_per_seed)),
            }
    return merged


def main():
    print("=" * 60)
    print("  SMC-QA 实验系统")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("\n[1/4] 加载数据...")
    raw_data = load_raw_data()
    print(f"  原始数据: {len(raw_data)} 条")

    dev_data, test_data = sample_data(raw_data)
    print(f"  开发集: {len(dev_data)} 条 | 测试集: {len(test_data)} 条")

    RESULTS_DIR.mkdir(exist_ok=True)

    if mode in ("dev", "all"):
        print("\n[2/4] 开发集实验...")
        dev_results = run_experiment(dev_data, METHODS, label="DEV", seed=42)
        save_results(dev_results, "dev_results.json")
        dev_summary = summarize_results(dev_results)
        print_table(dev_summary, "开发集结果")

    if mode in ("test", "all"):
        print("\n[3/4] 正式实验（3个随机种子）...")
        all_test_results = []
        for seed in EXPERIMENT_SEEDS:
            print(f"\n  --- Seed {seed} ---")
            _, test_split = sample_data(raw_data, seed=seed)
            r = run_experiment(test_split, METHODS, label=f"TEST-seed{seed}", seed=seed)
            save_results(r, f"test_results_seed{seed}.json")
            all_test_results.append(r)

        merged = merge_multi_seed_results(all_test_results)
        print_table(merged, "正式实验结果 (mean±std across 3 seeds)")
        with open(RESULTS_DIR / "test_merged.json", "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    if mode in ("ablation", "all"):
        print("\n[4/4] 消融实验...")
        combined_methods = {"SMC-QA (Ours)": METHODS["SMC-QA (Ours)"]}
        combined_methods.update(ABLATION_METHODS)
        abl_results = run_experiment(test_data, combined_methods, label="ABLATION")
        save_results(abl_results, "ablation_results.json")
        abl_summary = summarize_results(abl_results)
        print_table(abl_summary, "消融实验结果")

    print("\n全部实验完成！结果保存在 results/ 目录下。")


if __name__ == "__main__":
    main()
