# Improvement Code Summary

本文件概括 `Improvement/` 文件夹中的改进实验代码和结果组织方式。

## Folder Purpose

`Improvement/` 用于保存改进实验相关内容，和 `Note/` 分工如下：

| Folder | Purpose |
|--------|---------|
| `Note/` | 原始仓库复现记录、原始实验结果、复现日志 |
| `Improvement/` | 调参脚本、改进实验结果、改进日志 |

改进实验尽量不修改原仓库代码。所有参数变化都通过独立脚本在运行时临时注入，不写回 `src/`。

## Files

| File | Purpose |
|------|---------|
| `Improvement/RunLog.md` | 记录改进实验的运行环境、命令、搜索范围和结果摘要 |
| `Improvement/ChangeLog.md` | 记录改进文件夹中的新增、移动和实验发现 |
| `Improvement/CodeSummary.md` | 说明改进脚本结构和实验逻辑 |
| `Improvement/tune_smc.py` | 第一阶段：SMC-QA promotion 参数搜索 |
| `Improvement/tune_smc_retrieval.py` | 第二阶段：SMC-QA retrieval top-k 参数搜索 |
| `Improvement/results/` | 保存改进实验输出 |

## Shared Environment

改进实验沿用复现阶段的 conda 环境：

```bash
conda activate ml_project_repro
```

实际非交互运行可使用：

```bash
/Users/jermy/miniconda3/envs/ml_project_repro/bin/python Improvement/tune_smc.py
```

依赖版本保存在：

```text
Improvement/results/pip_freeze.txt
```

## Script 1: `tune_smc.py`

### Goal

搜索 SMC-QA 的 promotion 参数，看看是否可以提升 `Hit@6`、`Recall@6` 或 `Contains`。

### Tuned Parameters

| Parameter | Values |
|-----------|--------|
| promotion weights | baseline, rel_heavy, rel_strong, div_heavy |
| threshold | 0.25, 0.30, 0.35, 0.40 |
| promotion top-k | 3, 5 |
| route | auto, hybrid |

### Workflow

1. 加载 LongMemEval oracle 数据。
2. 使用原仓库 `sample_data()` 得到 dev/test split。
3. 对 dev set 预处理：切块、编码 embedding。
4. 在 dev set 上评估所有候选配置。
5. 选出 dev set 前 3 个配置。
6. 将前 3 个配置跑到 3 个 test seeds 上。
7. 保存 dev tuning 和 test merged 结果。

### Outputs

| File | Description |
|------|-------------|
| `Improvement/results/smc_tuning_dev_results.json` | 所有 dev 参数组合结果 |
| `Improvement/results/smc_tuned_test_results.json` | top dev configs 的 test 结果 |
| `Improvement/results/smc_tuning.log` | 控制台运行日志 |

## Script 2: `tune_smc_retrieval.py`

### Goal

测试 retrieval top-k 是否影响 SMC-QA 表现。

原始 router 中：

- `STM_TOP_K = 4`
- `LTM_TOP_K = 4`
- 指标是 `Hit@6` / `Recall@6`

因此有一个潜在问题：部分路由只返回 4 条，但指标评估 top 6。这个脚本在不修改 `src/router.py` 的前提下，运行时临时测试更宽的 STM/LTM retrieval depth。

### Tuned Parameters

| Parameter | Values |
|-----------|--------|
| STM top-k | 4, 6, 8 |
| LTM top-k | 4, 6, 8 |
| base SMC configs | first-stage top configs plus hybrid variant |

### Workflow

1. 复用 `tune_smc.py` 中的预处理和评估函数。
2. 临时 monkey patch `src.router.STM_TOP_K` 和 `src.router.LTM_TOP_K`。
3. 在 dev set 上评估 retrieval top-k configs。
4. 选出 dev set 前 3 个配置。
5. 在 3 个 test seeds 上评估。
6. 保存 tuning 和 test results。

### Outputs

| File | Description |
|------|-------------|
| `Improvement/results/smc_retrieval_tuning_dev_results.json` | retrieval top-k dev 搜索结果 |
| `Improvement/results/smc_retrieval_tuned_test_results.json` | retrieval top-k test 结果 |
| `Improvement/results/smc_retrieval_tuning.log` | 控制台运行日志 |

## Design Principle

改进脚本采用 wrapper 方式，而不是直接修改原始代码：

- import 原仓库模块；
- 复用原始 `pipeline._build_memory()`、`retrieve_with_routing()`、评测指标；
- 在运行时临时覆盖参数；
- 将结果写入 `Improvement/results/`；
- 不改动 `src/` 文件。

这样可以保证：

- 原始复现结果仍然可追溯；
- 改进实验可以独立说明；
- 报告中可以清楚区分 baseline reproduction 和 improvement attempt。

## Main Finding

第一阶段 promotion tuning 的收益较小；第二阶段 retrieval top-k tuning 更有效。

最佳观察配置：

```text
baseline_thr0.40_top3_auto_stm8_ltm8
```

含义：

- promotion weights 保持原始 baseline：relevance 0.4, reuse 0.4, diversity 0.2
- promotion threshold 提高到 0.40
- promotion top-k 保持 3
- routing 使用 auto
- STM retrieval top-k 从 4 增加到 8
- LTM retrieval top-k 从 4 增加到 8

测试集三 seeds merged 结果：

| Version | Hit@6 | Recall@6 | Contains |
|---------|-------|----------|----------|
| Original SMC-QA | 0.778 | 0.612 | 0.333 |
| Retrieval-tuned SMC-QA | 0.793 | 0.631 | 0.329 |

解释：

- 原始 SMC-QA 部分 route 只返回 4 条候选，但评估指标是 `@6`。
- 增大 STM/LTM route 的 retrieval depth 后，top-6 候选覆盖更充分。
- 改进主要体现在 retrieval coverage，不主要体现在 answer string containment。

## Supplementary Script: `supplementary_experiments.py`

### Goal

补充四类解释性实验：

1. question type breakdown
2. LTM quality analysis
3. retrieval depth ablation summary
4. tuned case studies

### Inputs

- LongMemEval oracle data
- seed 42 test split
- existing retrieval tuning results:
  - `smc_retrieval_tuning_dev_results.json`
  - `smc_retrieval_tuned_test_results.json`

### Compared Methods

| Method | Purpose |
|--------|---------|
| `Flat Memory` | non-hierarchical retrieval baseline |
| `Freq Promotion` | strongest baseline from original reproduction |
| `Original SMC-QA` | original project method |
| `Retrieval-tuned SMC-QA` | best improved variant |

### Outputs

| File | Meaning |
|------|---------|
| `question_type_breakdown_seed42.json` | per-question-type metrics and per-sample retrieval snippets |
| `ltm_quality_seed42.json` | LTM size, answer item rate, gold hit/recall in LTM, redundancy |
| `tuned_case_studies_seed42.json` | improved/regressed examples for tuned SMC-QA |
| `retrieval_depth_ablation_summary.json` | compact summary of retrieval depth tuning |
| `supplementary_summary.md` | human-readable report summary |

### Key Interpretation

- Retrieval tuning helps mainly by widening final candidate retrieval, not by making LTM itself better.
- Gains are clearest on `temporal-reasoning` and `single-session-assistant`.
- Some `knowledge-update` examples regress, suggesting future work should add conflict-aware or recency-aware routing.

## Optional Script: `tune_s_cleaned.py`

### Goal

专门针对 s_cleaned 大海捞针场景调优 SMC-QA。

### Why This Matters

原始复现中，s_cleaned 上 SMC-QA 不如 Flat Memory：

| Method | Hit@6 | Recall@6 |
|--------|------:|----------:|
| Flat Memory | 0.460 | 0.315 |
| Original SMC-QA | 0.420 | 0.290 |

因此 s_cleaned 是项目中最明显的短板。

### Search Space

| Parameter | Values |
|-----------|--------|
| weights | baseline, rel_heavy, div_heavy |
| threshold | 0.20, 0.30, 0.40 |
| promotion top-k | 3, 8 |
| STM/LTM top-k | 4/4, 8/8 |

### Output

| File | Meaning |
|------|---------|
| `s_cleaned_tuning_results.json` | tuning, validation, and full 100-item comparison |
| `s_cleaned_tuning.log` | console log |

### Key Result

Best full 100-item result:

| Method | Hit@6 | Recall@6 | Contains |
|--------|------:|----------:|---------:|
| Tuned SMC-QA | 0.510 | 0.360 | 0.270 |
| Flat Memory | 0.460 | 0.315 | 0.260 |
| Original SMC-QA | 0.420 | 0.290 | 0.250 |

## Optional Script: `answer_generation.py`

### Goal

测试 retrieval 改进是否能转化为下游 answer generation 改进。

### Modes

| Mode | Requirement | Output |
|------|-------------|--------|
| dry-run | no API key needed | `answer_generation_dry_run.json` |
| API run | `OPENAI_API_KEY` and `openai` package | `answer_generation_api.json` |

### Compared Retrieval Methods

- `Freq Promotion`
- `Original SMC-QA`
- `Retrieval-tuned SMC-QA`

### API Design

The script uses the OpenAI Responses API through:

```python
client.responses.create(...)
```

The model is controlled by `OPENAI_MODEL`; default is `gpt-5.2`.

### Current Status

The experiment has been run in dry-run mode:

- `Improvement/results/answer_generation_dry_run.json`
- `Improvement/results/answer_generation_dry_run.log`

No API calls were made because `OPENAI_API_KEY` was not set in the environment.
