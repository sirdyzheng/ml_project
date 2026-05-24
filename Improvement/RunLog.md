# Improvement Run Log

本文件记录在不修改原仓库源码的前提下，对 SMC-QA 进行改进实验的过程。

## 基本原则

- `Note/` 只用于原始复现记录。
- `Improvement/` 用于调参、改进脚本、改进结果和改进日志。
- 不修改 `src/`、`run_experiments.py`、`run_s_cleaned.py`、`run_case_analysis.py`。
- 所有改进实验通过 wrapper / monkey patch 在进程内临时修改参数，实验结束后不会写回原代码。

## 环境配置

沿用 `Note` 复现时创建的 conda 环境：

```bash
cd "/Users/MyDisk/E/ZMC/Course/机器学习 （MAT8034）/Final Project/ml_project"
conda activate ml_project_repro
python -m pip install sentence-transformers tqdm numpy
```

非交互 shell 中使用：

```bash
conda run -n ml_project_repro python -m pip install sentence-transformers tqdm numpy
```

依赖版本记录：

- `Improvement/results/pip_freeze.txt`

## 已执行改进实验

### 1. SMC Promotion 参数搜索

脚本：

```bash
conda run -n ml_project_repro python Improvement/tune_smc.py
```

实际运行：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/tune_smc.py \
  2>&1 | tee Improvement/results/smc_tuning.log
```

搜索范围：

- promotion weights:
  - baseline: relevance 0.4, reuse 0.4, diversity 0.2
  - rel_heavy: relevance 0.5, reuse 0.3, diversity 0.2
  - rel_strong: relevance 0.6, reuse 0.2, diversity 0.2
  - div_heavy: relevance 0.4, reuse 0.3, diversity 0.3
- threshold: 0.25, 0.30, 0.35, 0.40
- promotion top-k: 3, 5
- route: auto, hybrid

输出：

- `Improvement/results/smc_tuning_dev_results.json`
- `Improvement/results/smc_tuned_test_results.json`
- `Improvement/results/smc_tuning.log`

### 2. Retrieval Top-K 参数搜索

脚本：

```bash
conda run -n ml_project_repro python Improvement/tune_smc_retrieval.py
```

实际运行：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/tune_smc_retrieval.py \
  2>&1 | tee Improvement/results/smc_retrieval_tuning.log
```

输出：

- `Improvement/results/smc_retrieval_tuning_dev_results.json`
- `Improvement/results/smc_retrieval_tuned_test_results.json`
- `Improvement/results/smc_retrieval_tuning.log`

## 第一阶段结果摘要

单纯调整 promotion weights / threshold / promotion top-k 后，最好的 test 结果如下：

| Config | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| `baseline_thr0.40_top3_auto` | 0.776 +/- 0.023 | 0.614 +/- 0.016 | 0.336 +/- 0.035 |
| `baseline_thr0.25_top3_auto` | 0.776 +/- 0.017 | 0.609 +/- 0.013 | 0.333 +/- 0.030 |
| `div_heavy_thr0.40_top5_auto` | 0.760 +/- 0.016 | 0.593 +/- 0.016 | 0.333 +/- 0.036 |

观察：

- `baseline_thr0.40_top3_auto` 相比原始 `SMC-QA (Ours)` 的 `Contains` 略有提升。
- 仅调整 promotion 参数不能稳定超过 `Freq Promotion`。
- 下一步重点测试 retrieval top-k，因为原始 router 在 STM-only 或 LTM-only route 中只取 4 条，但指标是 Hit@6 / Recall@6。

## 第二阶段结果摘要

Retrieval top-k tuning 的 dev set 最优配置：

| Config | Dev Hit@6 | Dev Recall@6 | Dev Contains |
|--------|-----------|--------------|--------------|
| `baseline_thr0.40_top3_auto_stm6_ltm6` | 0.800 | 0.643 | 0.280 |
| `baseline_thr0.40_top3_auto_stm8_ltm8` | 0.800 | 0.643 | 0.280 |
| `div_heavy_thr0.40_top5_auto_stm6_ltm4` | 0.780 | 0.623 | 0.300 |

三组 test seeds 上的 merged 结果：

| Config | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| `baseline_thr0.40_top3_auto_stm6_ltm6` | 0.789 +/- 0.033 | 0.628 +/- 0.024 | 0.329 +/- 0.033 |
| `baseline_thr0.40_top3_auto_stm8_ltm8` | 0.793 +/- 0.030 | 0.631 +/- 0.024 | 0.329 +/- 0.033 |
| `div_heavy_thr0.40_top5_auto_stm6_ltm4` | 0.769 +/- 0.019 | 0.606 +/- 0.017 | 0.327 +/- 0.033 |

与原始复现结果对比：

| Method | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| Original SMC-QA | 0.778 | 0.612 | 0.333 |
| Best retrieval-tuned SMC-QA | 0.793 | 0.631 | 0.329 |
| Freq Promotion baseline | 0.804 | 0.641 | 0.327 |

结论：

- widening STM/LTM retrieval depth from 4 to 8 improves SMC-QA Hit@6 and Recall@6.
- best tuned SMC-QA is closer to `Freq Promotion` and slightly improves over original SMC-QA on retrieval coverage.
- `Contains` is roughly unchanged, so this改进主要体现在 retrieval quality，而不是 answer-string containment。

## 补充实验

### 3. Question Type Breakdown / LTM Quality / Case Study

脚本：

```bash
conda run -n ml_project_repro python Improvement/supplementary_experiments.py
```

实际运行：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/supplementary_experiments.py \
  2>&1 | tee Improvement/results/supplementary_experiments.log
```

输出：

- `Improvement/results/question_type_breakdown_seed42.json`
- `Improvement/results/ltm_quality_seed42.json`
- `Improvement/results/tuned_case_studies_seed42.json`
- `Improvement/results/retrieval_depth_ablation_summary.json`
- `Improvement/results/supplementary_summary.md`
- `Improvement/results/supplementary_experiments.log`

补充实验使用 seed 42 test split，目的是解释改进效果来自哪里，而不是替代三 seeds 的主结果。

### Question Type Breakdown Summary

Retrieval-tuned SMC-QA 相比 original SMC-QA：

| Question Type | Original Hit@6 | Tuned Hit@6 | Original Recall@6 | Tuned Recall@6 |
|---------------|---------------:|------------:|------------------:|---------------:|
| single-session-assistant | 0.588 | 0.647 | 0.588 | 0.647 |
| temporal-reasoning | 0.769 | 0.821 | 0.544 | 0.569 |
| multi-session | 0.750 | 0.722 | 0.485 | 0.501 |
| knowledge-update | 0.778 | 0.741 | 0.580 | 0.543 |

观察：

- retrieval tuning 对 `temporal-reasoning` 和 `single-session-assistant` 更有帮助。
- 对 `knowledge-update` 有回退，说明扩大检索深度可能引入较旧或冲突记忆。
- `single-session-user` 和 `single-session-preference` 基本不变。

### LTM Quality Summary

| Variant | LTM Size | Answer Item Rate | Gold Hit In LTM | Gold Recall In LTM |
|---------|---------:|-----------------:|----------------:|-------------------:|
| Freq Promotion | 8.7 | 0.144 | 0.787 | 0.607 |
| Original SMC-QA | 13.6 | 0.100 | 0.800 | 0.650 |
| Retrieval-tuned SMC-QA | 10.1 | 0.120 | 0.780 | 0.603 |

观察：

- Original SMC-QA 的 LTM 更大，gold recall in LTM 更高。
- Retrieval-tuned SMC-QA 的最终 retrieval 指标提升，主要来自更宽的 retrieval depth，而不是 LTM 本身质量提升。
- Freq Promotion 的 answer item rate 更高，这解释了它在 Hit@6 / Recall@6 上仍然强。

### Case Study Summary

Tuned SMC-QA 改善案例：

- `single-session-assistant`: Catalonia singer-songwriter question, recall 0.00 -> 1.00
- `multi-session`: tomatoes and cucumbers initial plants, recall 0.50 -> 1.00
- `temporal-reasoning`: charity gala vs charity bake sale, recall 0.00 -> 0.50

Tuned SMC-QA 回退案例：

- `knowledge-update`: old sneakers location, recall 1.00 -> 0.50
- `knowledge-update`: Alex from Germany meetups, recall 0.50 -> 0.00
- `multi-session`: car wash and parking ticket spending, recall 1.00 -> 0.50

## Optional Experiments

### 4. s_cleaned Promotion / Retrieval Tuning

脚本：

```bash
conda run -n ml_project_repro python Improvement/tune_s_cleaned.py
```

实际运行：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/tune_s_cleaned.py \
  2>&1 | tee Improvement/results/s_cleaned_tuning.log
```

设置：

- dataset: `longmemeval_s_cleaned.json`
- sample size: 100
- max turns per item: 200
- tuning split: first 30 sampled items
- validation split: remaining 70 sampled items
- tuned parameters:
  - promotion weights: baseline, rel_heavy, div_heavy
  - threshold: 0.20, 0.30, 0.40
  - promotion top-k: 3, 8
  - STM/LTM retrieval top-k: 4/4, 8/8

输出：

- `Improvement/results/s_cleaned_tuning_results.json`
- `Improvement/results/s_cleaned_tuning.log`

最佳验证配置：

```text
rel_heavy_thr0.20_top3_stm8_ltm8
```

Full 100-item s_cleaned comparison:

| Method | Hit@6 | Recall@6 | Contains |
|--------|------:|----------:|---------:|
| Tuned SMC-QA | 0.510 | 0.360 | 0.270 |
| Flat Memory | 0.460 | 0.315 | 0.260 |
| Freq Promotion | 0.430 | 0.292 | 0.250 |
| Original SMC-QA | 0.420 | 0.290 | 0.250 |

结论：

- s_cleaned 专项调优明显改善了 SMC-QA。
- 最有效变化是更宽的 retrieval top-k，同时使用更低 threshold 和 relevance-heavy promotion。
- 在这个 100-item s_cleaned sample 上，tuned SMC-QA 超过 Flat Memory、Freq Promotion 和 original SMC-QA。

### 5. Optional LLM Answer Generation

脚本：

```bash
conda run -n ml_project_repro python Improvement/answer_generation.py --n 20
```

实际 dry-run：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/answer_generation.py --n 20 \
  2>&1 | tee Improvement/results/answer_generation_dry_run.log
```

输出：

- `Improvement/results/answer_generation_dry_run.json`
- `Improvement/results/answer_generation_dry_run.log`

说明：

- 当前环境没有设置 `OPENAI_API_KEY`，所以没有实际调用 API。
- dry-run 已生成 20 条 seed-42 test prompts。
- 比较方法包括：
  - `Freq Promotion`
  - `Original SMC-QA`
  - `Retrieval-tuned SMC-QA`
- 默认模型来自 `OPENAI_MODEL`，未设置时使用 `gpt-5.2`。

实际运行 API 版本需要：

```bash
conda run -n ml_project_repro python -m pip install openai
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.2"
conda run -n ml_project_repro python Improvement/answer_generation.py --n 20 --run-api
```

API 输出将保存为：

```text
Improvement/results/answer_generation_api.json
```
