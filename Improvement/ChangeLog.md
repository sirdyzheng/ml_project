# Improvement Change Log

本文件记录改进实验相关的新增内容。`Note/` 保持为复现文件夹，改进内容统一放在 `Improvement/`。

## 2026-05-24

### Added

- Created `Improvement/` directory for improvement experiments.
- Created `Improvement/results/` for tuning outputs.
- Added `Improvement/tune_smc.py`.
  - Performs SMC-QA promotion parameter search on dev set.
  - Evaluates top dev configurations on three official test seeds.
  - Writes outputs only to `Improvement/results/`.
- Added `Improvement/tune_smc_retrieval.py`.
  - Tests widened STM/LTM retrieval depths without editing `src/router.py`.
  - Writes outputs only to `Improvement/results/`.
- Added `Improvement/results/pip_freeze.txt`.
  - Uses the same conda environment configuration as `Note`.

### Moved

- Moved improvement scripts and first-stage tuning outputs out of `Note/` into `Improvement/`.
- `Note/` is now reserved for original reproduction artifacts.

### Not Changed

- No changes to `src/`.
- No changes to `run_experiments.py`, `run_s_cleaned.py`, or `run_case_analysis.py`.
- No changes to original repository `results/`.

### First-Stage Finding

- Promotion-only tuning gives a small `Contains` improvement but does not materially improve `Hit@6` / `Recall@6`.
- Retrieval top-k tuning is added as the next improvement direction.

### Completed

- Ran `Improvement/tune_smc_retrieval.py`.
- Added retrieval tuning outputs:
  - `Improvement/results/smc_retrieval_tuning.log`
  - `Improvement/results/smc_retrieval_tuning_dev_results.json`
  - `Improvement/results/smc_retrieval_tuned_test_results.json`

### Second-Stage Finding

- Increasing STM/LTM retrieval depth improves SMC-QA retrieval metrics.
- Best observed configuration:
  - `baseline_thr0.40_top3_auto_stm8_ltm8`
  - `Hit@6 = 0.793 +/- 0.030`
  - `Recall@6 = 0.631 +/- 0.024`
  - `Contains = 0.329 +/- 0.033`
- Compared with original SMC-QA (`Hit@6 = 0.778`, `Recall@6 = 0.612`, `Contains = 0.333`), retrieval tuning improves Hit@6 and Recall@6 while keeping Contains similar.

## Supplementary Experiments

### Added

- Added `Improvement/supplementary_experiments.py`.
- Added supplementary result files:
  - `Improvement/results/question_type_breakdown_seed42.json`
  - `Improvement/results/ltm_quality_seed42.json`
  - `Improvement/results/tuned_case_studies_seed42.json`
  - `Improvement/results/retrieval_depth_ablation_summary.json`
  - `Improvement/results/supplementary_summary.md`
  - `Improvement/results/supplementary_experiments.log`

### Findings

- Question type breakdown shows retrieval tuning helps `temporal-reasoning` and `single-session-assistant` most clearly.
- Retrieval tuning regresses on some `knowledge-update` cases, likely because wider retrieval can surface conflicting older memories.
- LTM quality analysis shows original SMC-QA has higher gold recall inside LTM, while tuned SMC-QA improves final retrieval mainly through wider retrieval depth.
- Case studies now include examples where tuned SMC-QA improves over original SMC-QA and examples where it regresses.

## Optional Experiment Additions

### Added

- Added `Improvement/tune_s_cleaned.py`.
  - Tunes SMC-QA specifically for s_cleaned.
  - Writes `Improvement/results/s_cleaned_tuning_results.json`.
  - Writes `Improvement/results/s_cleaned_tuning.log`.
- Added `Improvement/answer_generation.py`.
  - Builds optional LLM answer generation prompts.
  - Supports OpenAI Responses API when `OPENAI_API_KEY` is set.
  - Writes dry-run prompts to `Improvement/results/answer_generation_dry_run.json`.
  - Writes actual API outputs to `Improvement/results/answer_generation_api.json` when `--run-api` is used.

### Results

- s_cleaned tuning best config:
  - `rel_heavy_thr0.20_top3_stm8_ltm8`
- Full 100-item s_cleaned results:
  - Tuned SMC-QA: `Hit@6 = 0.510`, `Recall@6 = 0.360`, `Contains = 0.270`
  - Flat Memory: `Hit@6 = 0.460`, `Recall@6 = 0.315`, `Contains = 0.260`
  - Freq Promotion: `Hit@6 = 0.430`, `Recall@6 = 0.292`, `Contains = 0.250`
  - Original SMC-QA: `Hit@6 = 0.420`, `Recall@6 = 0.290`, `Contains = 0.250`
- LLM answer generation was run in dry-run mode because `OPENAI_API_KEY` was not set.
