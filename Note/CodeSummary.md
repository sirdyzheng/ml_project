# Code Summary

本文件概括 SMC-QA 仓库的代码结构、实验入口和核心流程，方便后续复现与写报告。

## Project Goal

该项目研究长上下文问答中的记忆管理问题，比较多种从历史对话中检索相关信息的方法：

- `Flat Memory`
- `STM-only`
- `LTM-only`
- `Naive STM+LTM`
- `Freq Promotion`
- `SMC-QA (Ours)`

核心思想是使用短期记忆 STM、长期记忆 LTM、promotion 策略和 query routing 来提升长对话场景下的检索质量。

## Main Entrypoints

| File | Purpose |
|------|---------|
| `run_experiments.py` | 主实验入口，支持 `dev`、`test`、`ablation`、`all` 模式 |
| `run_s_cleaned.py` | 运行 s_cleaned 大海捞针实验 |
| `run_case_analysis.py` | 生成成功/失败案例分析 |

## Core Modules

| File | Responsibility |
|------|----------------|
| `src/config.py` | 集中管理路径、随机种子、切块参数、记忆容量、promotion 权重和检索参数 |
| `src/data_loader.py` | 加载 LongMemEval 数据、抽样 dev/test split、提取 session 和 gold turns |
| `src/chunking.py` | 将对话历史切成可检索的文本 chunk |
| `src/embedder.py` | 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成文本向量 |
| `src/memory.py` | 定义 `MemoryItem`、`FlatMemory`、`STM`、`LTM` 等记忆结构 |
| `src/promotion.py` | 实现 SMC promotion 和 frequency promotion |
| `src/router.py` | 根据 query 类型和相似度结果决定 STM/LTM/hybrid 检索路径 |
| `src/evaluate.py` | 计算 `Hit@6`、`Recall@6`、`Contains` 等指标 |
| `src/pipeline.py` | 串联数据预处理、向量化、各方法运行、评测与结果保存 |

## Experiment Flow

1. `run_experiments.py` 调用 `load_raw_data()` 读取 LongMemEval 数据。
2. `sample_data()` 切分 dev/test 数据。
3. `run_experiment()` 对每条样本执行：
   - `extract_sessions()` 提取会话历史。
   - `chunk_turns()` 将历史对话切成 chunks。
   - `encode_texts()` 批量生成 chunk 和 query embedding。
   - 对每个方法运行对应 runner。
   - 使用 `hit_at_k()`、`recall_at_k()`、`contains_match()` 评测。
4. `save_results()` 将逐样本指标写入 `results/*.json`。
5. `summarize_results()` 和 `print_table()` 汇总展示结果。

## Memory Methods

### Flat Memory

将所有 chunks 放入同一个 memory pool，根据 query embedding 做相似度检索。

### STM-only

只使用短期记忆，保留最近的 chunks，并从其中检索。

### LTM-only

通过 promotion 策略把部分 STM 内容提升到长期记忆，只从 LTM 中检索。

### Naive STM+LTM

同时使用 STM 和 LTM，但固定使用 hybrid 检索。

### Freq Promotion

根据 chunk 被命中的频率决定是否提升到 LTM。

### SMC-QA

综合 relevance、reuse 和 diversity 进行 promotion，并使用 query routing 自动决定检索路径。

## Key Hyperparameters

| Name | Value |
|------|-------|
| `CHUNK_SIZE` | 120 |
| `CHUNK_OVERLAP` | 30 |
| `STM_CAPACITY` | 40 |
| `LTM_MAX_SIZE` | 120 |
| `CONSOLIDATION_FREQ` | 5 |
| `PROMOTION_TOP_K` | 3 |
| `SCORE_THRESHOLD` | 0.35 |
| `DUPLICATE_SIM_THRESHOLD` | 0.85 |
| `RETRIEVAL_TOP_K` | 6 |
| `STM_TOP_K` | 4 |
| `LTM_TOP_K` | 4 |
| `SCORE_WEIGHTS` | relevance 0.4, reuse 0.4, diversity 0.2 |

## Result Files

| File | Meaning |
|------|---------|
| `results/dev_results.json` | 开发集实验结果 |
| `results/test_results_seed42.json` | seed 42 正式实验结果 |
| `results/test_results_seed43.json` | seed 43 正式实验结果 |
| `results/test_results_seed44.json` | seed 44 正式实验结果 |
| `results/test_merged.json` | 三个 seed 的均值和标准差 |
| `results/ablation_results.json` | 消融实验结果 |
| `results/s_cleaned_results.json` | s_cleaned 实验结果 |
| `results/case_analysis.json` | 案例分析结果 |

## Reproduction Notes

- `data/` 目录未上传到 git，需要单独下载。
- 首次运行会下载 sentence-transformers 模型，可能需要稳定网络。
- README 中预计 `run_experiments.py all` 约 10 分钟，`run_s_cleaned.py` 约 5 分钟，具体耗时取决于机器性能。
