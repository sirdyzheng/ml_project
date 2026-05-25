#!/usr/bin/env python3
"""Generate project figures from saved SMC-QA experiment results.

The script is intentionally self-contained inside Figure/. It reads JSON files
from ../results and writes both PNG and PDF outputs to Figure/output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "output"

PALETTE = {
    "blue_main": "#599CB4",
    "blue_soft": "#88CEE6",
    "green_main": "#7C9895",
    "green_soft": "#C9DCC4",
    "red_main": "#C25759",
    "red_soft": "#E89DA0",
    "violet_main": "#B696B6",
    "violet_soft": "#D9B9D4",
    "teal": "#7C9895",
    "neutral": "#AEB2D1",
    "warm": "#DAA87C",
    "axis": "#767676",
    "grid": "#E2E5EA",
    "text": "#303030",
}

METRIC_STYLES = [
    ("hit@6", "Hit@6", PALETTE["blue_soft"], PALETTE["blue_main"]),
    ("recall@6", "Recall@6", PALETTE["green_soft"], PALETTE["green_main"]),
    ("contains", "Contains", PALETTE["violet_soft"], PALETTE["violet_main"]),
]

METRICS = [
    ("hit@6", "Hit@6"),
    ("recall@6", "Recall@6"),
    ("contains", "Contains"),
]

LEGEND_FONT_SIZE = 12.0
ANNOTATION_FONT_SIZE = 9.2


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def metric_mean(payload: dict, metric: str) -> float:
    value = payload[metric]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def metric_std(payload: dict, metric: str) -> float:
    value = payload[metric]
    if isinstance(value, dict):
        return float(value.get("std", 0.0))
    return 0.0


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "font.weight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 180,
        }
    )


def finish_axes(
    ax: plt.Axes,
    ylim: tuple[float, float] = (0, 0.9),
    *,
    y_major: float = 0.1,
    grid_axis: str = "y",
) -> None:
    ax.set_ylim(*ylim)
    ax.grid(axis=grid_axis, which="major", color=PALETTE["grid"], linestyle="--", dashes=(3, 3), linewidth=0.9, zorder=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="both", colors=PALETTE["text"], width=1.0, length=4)
    if grid_axis == "y":
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", colors=PALETTE["axis"], width=0.8, length=2.5)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color(PALETTE["text"])


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=360, pad_inches=0.08)
    plt.close(fig)
    print(f"saved {OUT_DIR / (stem + '.png')}")


def add_top_legend(ax: plt.Axes, *, ncols: int, y: float = 1.005, fontsize: float = LEGEND_FONT_SIZE) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncols=ncols,
        fontsize=fontsize,
        handlelength=1.8,
        columnspacing=1.25,
        handletextpad=0.5,
        borderaxespad=0.0,
        prop={"weight": "bold", "size": fontsize},
    )


def annotate_bars(
    ax: plt.Axes,
    bars,
    *,
    errors: np.ndarray | None = None,
    fmt: str = "{:.3f}",
    fontsize: float = ANNOTATION_FONT_SIZE,
    padding: float = 3.0,
) -> None:
    for idx, patch in enumerate(bars.patches):
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        error = 0.0 if errors is None else float(errors[idx])
        ax.annotate(
            fmt.format(height),
            xy=(patch.get_x() + patch.get_width() / 2.0, height + error),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PALETTE["text"],
            fontweight="bold",
        )


def grouped_metric_bar(
    ax: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    errors: np.ndarray | None = None,
    *,
    annotate: bool = False,
    ylim: tuple[float, float] = (0, 0.9),
    label_rotation: float = 0,
    legend_fontsize: float = LEGEND_FONT_SIZE,
    legend_y: float = 1.005,
) -> None:
    x = np.arange(len(labels))
    width = min(0.72 / len(METRIC_STYLES), 0.24)
    for idx, (metric_key, metric_label, face, edge) in enumerate(METRIC_STYLES):
        offset = (idx - 1) * width
        err = errors[:, idx] if errors is not None else None
        bars = ax.bar(
            x + offset,
            values[:, idx],
            width=width * 0.94,
            label=metric_label,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            yerr=err,
            capsize=3 if errors is not None else 0,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": PALETTE["axis"]},
            zorder=3,
        )
        if annotate:
            annotate_bars(ax, bars, errors=err)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelrotation=label_rotation, pad=3)
    ax.set_ylabel("Score", fontsize=12.5, labelpad=4)
    add_top_legend(ax, ncols=3, y=legend_y, fontsize=legend_fontsize)
    finish_axes(ax, ylim=ylim)


def fig01_oracle_main_results() -> None:
    data = load_json(RESULTS_DIR / "test_merged.json")
    labels = ["STM-only", "LTM-only", "Flat", "Naive", "Freq", "SMC-QA"]
    keys = ["STM-only", "LTM-only", "Flat Memory", "Naive STM+LTM", "Freq Promotion", "SMC-QA (Ours)"]
    values = np.array([[metric_mean(data[key], metric) for metric, _ in METRICS] for key in keys])
    errors = np.array([[metric_std(data[key], metric) for metric, _ in METRICS] for key in keys])

    fig, ax = plt.subplots(figsize=(7.0, 3.55), layout="none")
    grouped_metric_bar(ax, labels, values, errors, ylim=(0.25, 0.86), legend_fontsize=13.2, legend_y=0.985)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.15)
    save_figure(fig, "fig01_oracle_main_results")


def fig02_retrieval_tuning() -> None:
    base = load_json(RESULTS_DIR / "test_merged.json")["SMC-QA (Ours)"]
    tuned = load_json(RESULTS_DIR / "smc_retrieval_tuned_test_results.json")["merged"][
        "baseline_thr0.40_top3_auto_stm8_ltm8"
    ]
    freq = load_json(RESULTS_DIR / "test_merged.json")["Freq Promotion"]
    labels = ["Base SMC-QA", "Retrieval-tuned", "Freq Promotion"]
    rows = [base, tuned, freq]
    values = np.array([[metric_mean(row, metric) for metric, _ in METRICS] for row in rows])
    errors = np.array([[metric_std(row, metric) for metric, _ in METRICS] for row in rows])

    fig, ax = plt.subplots(figsize=(5.3, 3.45), layout="none")
    grouped_metric_bar(ax, labels, values, errors, annotate=True, ylim=(0.25, 0.86), label_rotation=0)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.16)
    save_figure(fig, "fig02_retrieval_tuning")


def fig03_s_cleaned_tuning() -> None:
    data = load_json(RESULTS_DIR / "s_cleaned_tuning_results.json")["full_sample"]
    labels = ["Tuned SMC-QA", "Flat", "Freq", "Base SMC-QA"]
    keys = ["Tuned SMC-QA", "Flat Memory", "Freq Promotion", "Base SMC-QA"]
    values = np.array([[metric_mean(data[key]["summary"], metric) for metric, _ in METRICS] for key in keys])

    fig, ax = plt.subplots(figsize=(5.8, 3.45), layout="none")
    grouped_metric_bar(ax, labels, values, annotate=True, ylim=(0.20, 0.55), label_rotation=0)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.16)
    save_figure(fig, "fig03_s_cleaned_tuning")


def fig04_question_type_delta() -> None:
    data = load_json(RESULTS_DIR / "question_type_breakdown_seed42.json")["by_type"]
    base = data["Base SMC-QA"]
    tuned = data["Retrieval-tuned SMC-QA"]
    qtypes = sorted(set(base) & set(tuned))
    hit_delta = np.array([tuned[q]["hit@6"] - base[q]["hit@6"] for q in qtypes])
    recall_delta = np.array([tuned[q]["recall@6"] - base[q]["recall@6"] for q in qtypes])
    display = [q.replace("single-session-", "single-").replace("-", "\n") for q in qtypes]

    y = np.arange(len(qtypes))
    fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="none")
    ax.axvline(0, color=PALETTE["axis"], linewidth=1.2, zorder=1)
    bars1 = ax.barh(y + 0.18, hit_delta, height=0.34, color=PALETTE["blue_soft"], edgecolor=PALETTE["blue_main"], linewidth=1.0, label="Hit@6 delta", zorder=3)
    bars2 = ax.barh(y - 0.18, recall_delta, height=0.34, color=PALETTE["green_soft"], edgecolor=PALETTE["green_main"], linewidth=1.0, label="Recall@6 delta", zorder=3)
    ax.set_yticks(y, display)
    ax.set_xlabel("Retrieval-tuned minus Base SMC-QA", fontsize=12, labelpad=4)
    # The paper caption provides the title; keep the panel itself compact.
    add_top_legend(ax, ncols=2)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", dashes=(3, 3), linewidth=0.9, zorder=0)
    ax.tick_params(axis="both", colors=PALETTE["text"], width=1.0, length=4)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.set_xlim(-0.055, 0.075)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.025))
    for bars in [bars1, bars2]:
        for patch in bars:
            val = patch.get_width()
            ax.text(
                val + (0.003 if val >= 0 else -0.003),
                patch.get_y() + patch.get_height() / 2,
                f"{val:+.3f}",
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=PALETTE["text"],
                fontweight="bold",
            )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.82, bottom=0.16)
    save_figure(fig, "fig04_question_type_delta")


def fig05_ltm_quality() -> None:
    data = load_json(RESULTS_DIR / "ltm_quality_seed42.json")["overall"]
    labels = ["Freq", "Base SMC-QA", "Retrieval-tuned"]
    keys = ["Freq Promotion", "Base SMC-QA", "Retrieval-tuned SMC-QA"]
    ltm_size = [data[key]["ltm_size"] for key in keys]
    answer_rate = [data[key]["answer_item_rate"] for key in keys]
    gold_hit = [data[key]["gold_hit_in_ltm"] for key in keys]
    gold_recall = [data[key]["gold_recall_in_ltm"] for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), layout="none", gridspec_kw={"width_ratios": [0.88, 1.32]})
    axes[0].bar(labels, ltm_size, color=[PALETTE["violet_soft"], PALETTE["blue_soft"], PALETTE["green_soft"]], edgecolor=[PALETTE["violet_main"], PALETTE["blue_main"], PALETTE["green_main"]], linewidth=1.0)
    axes[0].set_title("Average LTM Size", fontsize=13, pad=10)
    axes[0].set_ylabel("Chunks", fontsize=12)
    finish_axes(axes[0], ylim=(0, 15), y_major=5)
    axes[0].tick_params(axis="x", rotation=16)
    for idx, value in enumerate(ltm_size):
        axes[0].text(
            idx,
            value + 0.35,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=PALETTE["text"],
        )

    x = np.arange(len(labels))
    width = 0.24
    series = [
        ("Answer item rate", answer_rate, PALETTE["blue_soft"], PALETTE["blue_main"]),
        ("Gold hit in LTM", gold_hit, PALETTE["green_soft"], PALETTE["green_main"]),
        ("Gold recall in LTM", gold_recall, PALETTE["violet_soft"], PALETTE["violet_main"]),
    ]
    for idx, (name, vals, face, edge) in enumerate(series):
        axes[1].bar(x + (idx - 1) * width, vals, width=width * 0.94, label=name, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=3)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("LTM Evidence Quality", fontsize=13, pad=10)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].tick_params(axis="x", rotation=16)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=3,
        prop={"weight": "bold", "size": 9.4},
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    finish_axes(axes[1], ylim=(0, 0.88), y_major=0.2)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.28, wspace=0.33)
    save_figure(fig, "fig05_ltm_quality")


def main() -> None:
    apply_style()
    fig01_oracle_main_results()
    fig02_retrieval_tuning()
    fig03_s_cleaned_tuning()
    fig04_question_type_delta()
    fig05_ltm_quality()
    print(f"\nAll figures saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
