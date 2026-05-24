#!/usr/bin/env python3
"""Optional LLM answer generation experiment.

This script is intentionally optional. Without OPENAI_API_KEY it creates a
dry-run prompt dataset. With OPENAI_API_KEY and the openai package installed,
it calls the OpenAI Responses API and evaluates generated answers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from src.evaluate import contains_match, exact_match, hit_at_k, normalize_text, recall_at_k
from tune_smc import temporary_pipeline_globals

RESULTS_DIR = PROJECT_ROOT / "Improvement" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TUNED_CONFIG = {
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


def preprocess(data_items: List[Dict[str, Any]]):
    all_texts: List[str] = []
    ranges = []
    for item in tqdm(data_items, desc="answer gen chunking"):
        turns = extract_sessions(item)
        chunks = chunk_turns(turns)
        start = len(all_texts)
        all_texts.extend([chunk["text"] for chunk in chunks])
        all_texts.append(item["question"])
        ranges.append((start, len(all_texts) - 1, chunks, item))

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
    with temporary_router_globals(STM_TOP_K=TUNED_CONFIG["stm_top_k"], LTM_TOP_K=TUNED_CONFIG["ltm_top_k"]):
        return router.retrieve_with_routing(
            sample["question"],
            sample["query_emb"],
            stm,
            ltm,
            route=TUNED_CONFIG["route"],
        )


def retrieve(sample: Dict[str, Any], method: str):
    if method == "Flat Memory":
        return pipeline.run_flat(sample["chunks"], sample["query_emb"])
    if method == "Freq Promotion":
        return pipeline.run_freq_promotion(sample["chunks"], sample["query_emb"], query_text=sample["question"])
    if method == "Original SMC-QA":
        return pipeline.run_smc_qa(sample["chunks"], sample["query_emb"], query_text=sample["question"])
    if method == "Retrieval-tuned SMC-QA":
        return run_tuned_smc(sample)
    raise ValueError(method)


def build_prompt(sample: Dict[str, Any], retrieved) -> str:
    context_blocks = []
    for idx, item in enumerate(retrieved[:6], start=1):
        context_blocks.append(f"[{idx}] {item.text}")
    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, answer with 'I don't know'. "
        "Keep the answer concise.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def get_response_text(response) -> str:
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


def call_openai(client, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    return get_response_text(response)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of seed-42 test items to use.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Freq Promotion", "Original SMC-QA", "Retrieval-tuned SMC-QA"],
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--run-api", action="store_true", help="Actually call OpenAI API.")
    args = parser.parse_args()

    raw_data = load_raw_data()
    _, test_data = sample_data(raw_data, seed=42)
    data_items = test_data[: args.n]
    preprocessed = preprocess(data_items)

    can_call_api = args.run_api and bool(os.environ.get("OPENAI_API_KEY"))
    client = None
    if can_call_api:
        try:
            from openai import OpenAI

            client = OpenAI()
        except Exception as exc:
            print(f"OpenAI client unavailable; falling back to dry run: {exc}")
            can_call_api = False
    elif args.run_api:
        print("OPENAI_API_KEY is not set; falling back to dry run.")

    results = {
        "settings": {
            "n": args.n,
            "methods": args.methods,
            "model": args.model,
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
        for method in args.methods:
            retrieved = retrieve(sample, method)
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
                    generated = call_openai(client, args.model, prompt)
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
        for method in args.methods:
            method_rows = [item["methods"][method] for item in results["items"]]
            summary[method] = {}
            for metric in ["contains", "exact_match", "token_overlap", "latency"]:
                values = [
                    row["answer_metrics"][metric]
                    for row in method_rows
                    if metric in row["answer_metrics"]
                ]
                summary[method][metric] = float(np.mean(values)) if values else 0.0
        results["summary"] = summary
    else:
        results["summary"] = {
            "dry_run": "No API calls were made. Prompts are saved for later execution."
        }

    suffix = "api" if can_call_api else "dry_run"
    out_path = RESULTS_DIR / f"answer_generation_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
