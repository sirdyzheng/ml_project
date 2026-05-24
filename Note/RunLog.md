# Run Log

本文件用于记录 SMC-QA 项目的本地复现过程、运行命令、结果文件和遇到的问题。

## 基本信息

- Project: SMC-QA: Selective Memory Consolidation for Long-Context QA
- Repository: `https://github.com/sirdyzheng/ml_project.git`
- Local path: `Final Project/ml_project`
- Created on: 2026-05-19
- Dataset: LongMemEval oracle + s_cleaned

## 环境准备

推荐使用 conda 环境复现：

```bash
cd "/Users/MyDisk/E/ZMC/Course/机器学习 （MAT8034）/Final Project/ml_project"
conda create -n ml_project_repro python=3.11 -y
conda activate ml_project_repro
python -m pip install sentence-transformers tqdm numpy
```

本次实际运行时，为了在非交互 shell 中避免 `conda activate` 的状态问题，使用了等价命令：

```bash
conda create -n ml_project_repro python=3.11 -y
conda run -n ml_project_repro python -m pip install sentence-transformers tqdm numpy
```

备用方案（不推荐优先使用，仅用于没有 conda 时）：

```bash
cd "/Users/MyDisk/E/ZMC/Course/机器学习 （MAT8034）/Final Project/ml_project"
python -m venv .venv
source .venv/bin/activate
pip install sentence-transformers tqdm numpy
```

## 数据准备

仓库的 `data/` 目录没有上传到 git，需要本地下载数据。

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ..
```

如果本机没有 `wget`，可以改用：

```bash
curl -L -o data/longmemeval_oracle.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
curl -L -o data/longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

## 复现实验命令

### 1. 开发集实验

```bash
python run_experiments.py dev
```

输出文件：

- `results/dev_results.json`

### 2. 正式实验

```bash
python run_experiments.py test
```

输出文件：

- `results/test_results_seed42.json`
- `results/test_results_seed43.json`
- `results/test_results_seed44.json`
- `results/test_merged.json`

### 3. 消融实验

```bash
python run_experiments.py ablation
```

输出文件：

- `results/ablation_results.json`

### 4. 一键全量实验

```bash
python run_experiments.py all
```

### 5. S-Cleaned 实验

```bash
python run_s_cleaned.py
```

输出文件：

- `results/s_cleaned_results.json`

### 6. 案例分析

```bash
python run_case_analysis.py
```

输出文件：

- `results/case_analysis.json`

## 运行记录

| Date | Command | Status | Output / Note |
|------|---------|--------|---------------|
| 2026-05-19 | `git clone https://github.com/sirdyzheng/ml_project.git` | Success | SSH 无权限，改用 HTTPS clone 成功 |
| 2026-05-19 | `conda create -n ml_project_repro python=3.11 -y` | Success | 创建 conda 复现环境 |
| 2026-05-19 | `conda run -n ml_project_repro python -m pip install sentence-transformers tqdm numpy` | Success | 安装项目运行依赖 |
| 2026-05-19 | `hf_hub_download(..., local_dir="data")` | Success | 下载 `longmemeval_oracle.json` 和 `longmemeval_s_cleaned.json` |
| 2026-05-19 | `python -c` wrapper for `run_experiments.py all` | Success | 结果保存到 `Note/results/`，日志为 `run_experiments_all.log` |
| 2026-05-19 | `python -c` wrapper for `run_s_cleaned.py` | Success | 结果保存到 `Note/results/s_cleaned_results.json` |
| 2026-05-19 | `python -c` wrapper for `run_case_analysis.py` | Success | 结果保存到 `Note/results/case_analysis.json` |

说明：为避免改动原仓库源码，本次没有修改 `src/config.py` 的 `RESULTS_DIR`。运行时通过一次性 Python wrapper 在内存中把 `RESULTS_DIR` 指向 `Note/results/`。

## 本次复现输出

| File | Description |
|------|-------------|
| `Note/results/dev_results.json` | 开发集逐样本结果 |
| `Note/results/test_results_seed42.json` | seed 42 正式测试逐样本结果 |
| `Note/results/test_results_seed43.json` | seed 43 正式测试逐样本结果 |
| `Note/results/test_results_seed44.json` | seed 44 正式测试逐样本结果 |
| `Note/results/test_merged.json` | 三个 seed 的均值和标准差 |
| `Note/results/ablation_results.json` | 消融实验逐样本结果 |
| `Note/results/s_cleaned_results.json` | s_cleaned 实验逐样本结果 |
| `Note/results/case_analysis.json` | 案例分析结果 |
| `Note/results/run_experiments_all.log` | `dev/test/ablation` 控制台日志 |
| `Note/results/run_s_cleaned.log` | s_cleaned 控制台日志 |
| `Note/results/run_case_analysis.log` | 案例分析控制台日志 |
| `Note/results/pip_freeze.txt` | conda 环境中的 pip 依赖版本 |

## 关键复现结果

### Oracle Test Mean +/- Std

| Method | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| Flat Memory | 0.782 +/- 0.019 | 0.623 +/- 0.021 | 0.298 +/- 0.049 |
| STM-only | 0.642 +/- 0.003 | 0.478 +/- 0.002 | 0.342 +/- 0.042 |
| LTM-only | 0.713 +/- 0.019 | 0.549 +/- 0.014 | 0.309 +/- 0.044 |
| Naive STM+LTM | 0.776 +/- 0.017 | 0.614 +/- 0.018 | 0.329 +/- 0.039 |
| Freq Promotion | 0.804 +/- 0.013 | 0.641 +/- 0.021 | 0.327 +/- 0.045 |
| SMC-QA (Ours) | 0.778 +/- 0.022 | 0.612 +/- 0.018 | 0.333 +/- 0.030 |

### S-Cleaned

| Method | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| Flat Memory | 0.460 | 0.315 | 0.260 |
| STM-only | 0.050 | 0.033 | 0.070 |
| LTM-only | 0.420 | 0.290 | 0.270 |
| Naive STM+LTM | 0.420 | 0.290 | 0.250 |
| Freq Promotion | 0.430 | 0.292 | 0.250 |
| SMC-QA (Ours) | 0.420 | 0.290 | 0.250 |

## 待确认事项

- [x] 本地 Python 版本：conda `ml_project_repro` 使用 Python 3.11.15
- [x] 是否已创建 conda 环境：`ml_project_repro`
- [x] 是否已下载 `data/longmemeval_oracle.json`
- [x] 是否已下载 `data/longmemeval_s_cleaned.json`
- [x] 是否成功运行 `python run_experiments.py dev`
- [x] 是否成功复现 README 中的正式实验结果
