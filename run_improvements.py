#!/usr/bin/env python3
"""Run SMC-QA improvement and supplementary experiments.

Examples:
  python run_improvements.py retrieval
  python run_improvements.py supplementary
  python run_improvements.py s-cleaned
  python run_improvements.py answer-generation --n 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RESULTS_DIR
from src.data_loader import load_raw_data, sample_data
from src.improvement import (
    evaluate_methods,
    evaluate_config,
    evaluate_test_seeds,
    merge_seed_summaries,
    preprocess_items,
    promotion_config_grid,
    question_type_breakdown,
    retrieval_config_grid,
    retrieval_depth_ablation_summary,
    run_answer_generation,
    s_cleaned_config_grid,
    sample_s_cleaned,
    save_json,
    tune_configs,
    tuned_case_studies,
    ltm_quality,
    write_markdown_summary,
)


def print_top_configs(rows, limit=10):
    for rank, row in enumerate(rows[:limit], start=1):
        summary = row["summary"]
        print(
            f"{rank:>2}. {row['config']['name']}: "
            f"Hit@6={summary['hit@6']:.3f} "
            f"Recall@6={summary['recall@6']:.3f} "
            f"Contains={summary['contains']:.3f} "
            f"Objective={summary['objective']:.3f}"
        )


def run_promotion(args):
    raw_data = load_raw_data()
    dev_data, _ = sample_data(raw_data)
    dev_preprocessed = preprocess_items(dev_data, "PROMOTION DEV")

    tuning_rows = tune_configs(dev_preprocessed, promotion_config_grid(), "promotion tuning")
    save_json(tuning_rows, "smc_tuning_dev_results.json")

    print("\nTop promotion-only dev configs:")
    print_top_configs(tuning_rows)

    top_configs = [row["config"] for row in tuning_rows[: args.top_configs]]
    test_runs = evaluate_test_seeds(raw_data, top_configs, "PROMOTION TEST")
    merged = merge_seed_summaries(test_runs)
    save_json({"test_runs": test_runs, "merged": merged}, "smc_tuned_test_results.json")

    print("\nMerged promotion-only test summaries:")
    for name, metrics in merged.items():
        print(
            f"{name}: Hit@6={metrics['hit@6']['mean']:.3f}+/-{metrics['hit@6']['std']:.3f} "
            f"Recall@6={metrics['recall@6']['mean']:.3f}+/-{metrics['recall@6']['std']:.3f} "
            f"Contains={metrics['contains']['mean']:.3f}+/-{metrics['contains']['std']:.3f}"
        )


def run_retrieval(args):
    raw_data = load_raw_data()
    dev_data, _ = sample_data(raw_data)
    dev_preprocessed = preprocess_items(dev_data, "RETRIEVAL DEV")

    dev_rows = tune_configs(dev_preprocessed, retrieval_config_grid(), "retrieval top-k tuning")
    save_json(dev_rows, "smc_retrieval_tuning_dev_results.json")

    print("\nTop retrieval-depth dev configs:")
    print_top_configs(dev_rows)

    top_configs = [row["config"] for row in dev_rows[: args.top_configs]]
    test_runs = evaluate_test_seeds(raw_data, top_configs, "RETRIEVAL TEST")
    merged = merge_seed_summaries(test_runs)
    save_json({"test_runs": test_runs, "merged": merged}, "smc_retrieval_tuned_test_results.json")

    print("\nMerged retrieval-tuned test summaries:")
    for name, metrics in merged.items():
        print(
            f"{name}: Hit@6={metrics['hit@6']['mean']:.3f}+/-{metrics['hit@6']['std']:.3f} "
            f"Recall@6={metrics['recall@6']['mean']:.3f}+/-{metrics['recall@6']['std']:.3f} "
            f"Contains={metrics['contains']['mean']:.3f}+/-{metrics['contains']['std']:.3f}"
        )


def run_supplementary(args):
    raw_data = load_raw_data()
    _, test_data = sample_data(raw_data, seed=args.seed)
    preprocessed = preprocess_items(test_data, f"SUPPLEMENTARY TEST seed {args.seed}")

    qtype = question_type_breakdown(preprocessed)
    save_json(qtype, f"question_type_breakdown_seed{args.seed}.json")

    ltm = ltm_quality(preprocessed)
    save_json(ltm, f"ltm_quality_seed{args.seed}.json")

    cases = tuned_case_studies(qtype)
    save_json(cases, f"tuned_case_studies_seed{args.seed}.json")

    depth = retrieval_depth_ablation_summary()
    save_json(depth, "retrieval_depth_ablation_summary.json")

    summary_path = write_markdown_summary(qtype, ltm, cases, depth)
    print(f"\nSupplementary analyses saved under {RESULTS_DIR}")
    print(f"Markdown summary: {summary_path}")


def run_s_cleaned(args):
    data_items = sample_s_cleaned(sample_size=args.sample_size, seed=args.seed)
    preprocessed = preprocess_items(data_items, "S-CLEANED", max_turns=args.max_turns)
    tuning = preprocessed[: args.tuning_size]
    validation = preprocessed[args.tuning_size :]

    tuning_rows = tune_configs(tuning, s_cleaned_config_grid(), "s_cleaned tuning")
    save_json(tuning_rows, "s_cleaned_tuning_dev_results.json")

    print("\nTop s_cleaned tuning configs:")
    print_top_configs(tuning_rows)

    top_configs = [row["config"] for row in tuning_rows[: args.s_cleaned_top_configs]]
    validation_results = {}
    for config in top_configs:
        summary, metrics = evaluate_config(validation, config)
        validation_results[config["name"]] = {"config": config, "summary": summary, "metrics": metrics}
        print(
            f"[validation] {config['name']}: "
            f"Hit@6={summary['hit@6']:.3f} "
            f"Recall@6={summary['recall@6']:.3f} "
            f"Contains={summary['contains']:.3f}"
        )

    best_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["summary"]["objective"],
            validation_results[name]["summary"]["hit@6"],
            validation_results[name]["summary"]["recall@6"],
        ),
    )
    best_config = validation_results[best_name]["config"]
    best_summary, best_metrics = evaluate_config(preprocessed, best_config)
    baselines = evaluate_methods(preprocessed, ["Flat Memory", "Freq Promotion", "Base SMC-QA"])

    output = {
        "settings": {
            "sample_size": args.sample_size,
            "tuning_size": args.tuning_size,
            "validation_size": args.sample_size - args.tuning_size,
            "max_turns": args.max_turns,
            "seed": args.seed,
        },
        "tuning_results": tuning_rows,
        "validation_results": validation_results,
        "best_config": best_config,
        "full_sample": {"Tuned SMC-QA": {"summary": best_summary, "metrics": best_metrics}, **baselines},
    }
    save_json(output, "s_cleaned_tuning_results.json")

    print("\nFull s_cleaned comparison:")
    for name, payload in output["full_sample"].items():
        summary = payload["summary"]
        print(
            f"{name}: Hit@6={summary['hit@6']:.3f} "
            f"Recall@6={summary['recall@6']:.3f} "
            f"Contains={summary['contains']:.3f}"
        )


def run_answer_generation_cli(args):
    raw_data = load_raw_data()
    _, test_data = sample_data(raw_data, seed=args.seed)
    data_items = test_data[: args.n]
    preprocessed = preprocess_items(data_items, f"ANSWER GENERATION seed {args.seed}")

    results = run_answer_generation(
        preprocessed,
        methods=args.methods,
        model=args.model,
        run_api=args.run_api,
    )
    suffix = "api" if results["settings"]["run_api"] else "dry_run"
    out_path = save_json(results, f"answer_generation_{suffix}.json")
    print(f"Saved {out_path}")


def run_all(args):
    run_promotion(args)
    run_retrieval(args)
    run_supplementary(args)
    run_s_cleaned(args)
    print("\nAll non-API improvement experiments finished.")


def build_parser():
    parser = argparse.ArgumentParser(description="Run SMC-QA improvement experiments.")
    parser.add_argument("--top-configs", type=int, default=3, help="Number of top configs to evaluate on test/validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--tuning-size", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--s-cleaned-top-configs",
        type=int,
        default=5,
        help="Number of S-Cleaned configs to validate after tuning.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("promotion", help="Run promotion-only tuning.")
    subparsers.add_parser("retrieval", help="Run retrieval top-k tuning.")
    subparsers.add_parser("supplementary", help="Run question type, LTM quality, and case analyses.")
    subparsers.add_parser("s-cleaned", help="Run S-Cleaned tuning and comparison.")

    answer = subparsers.add_parser("answer-generation", help="Build prompts or run optional LLM answer generation.")
    answer.add_argument("--n", type=int, default=20)
    answer.add_argument(
        "--methods",
        nargs="+",
        default=["Freq Promotion", "Base SMC-QA", "Retrieval-tuned SMC-QA"],
    )
    answer.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    answer.add_argument("--run-api", action="store_true")

    subparsers.add_parser("all", help="Run promotion, retrieval, supplementary, and S-Cleaned experiments.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "promotion":
        run_promotion(args)
    elif args.command == "retrieval":
        run_retrieval(args)
    elif args.command == "supplementary":
        run_supplementary(args)
    elif args.command == "s-cleaned":
        run_s_cleaned(args)
    elif args.command == "answer-generation":
        run_answer_generation_cli(args)
    elif args.command == "all":
        run_all(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
