# SMC-QA: Selective Memory Consolidation for Long-Context QA

> MAT8034 Final Project — Category 2: Applied / Engineering
>
> **Zheng Yifan** (12532245) · **Chen Zeming** (12531410)

## 这个项目是做什么的

简单说：我们在研究**"怎么帮 AI 管理长对话记忆"**。

想象 AI 和人聊了几百轮，现在用户问了一个关于之前聊过的内容的问题。AI 要从几百条历史消息里找到答案。我们比较了 6 种不同的"翻笔记策略"：

| 方法 | 通俗解释 |
|------|----------|
| Flat Memory | 所有笔记堆一起搜 |
| STM-only | 只翻最近几页 |
| LTM-only | 只翻整理过的重点笔记 |
| Naive STM+LTM | 最近的和重点的都翻，但不想先查哪边 |
| Freq Promotion | 被翻到次数多的就算重点 |
| **SMC-QA (Ours)** | 综合"相关性+使用频率+不重复"来决定什么算重点，根据问题类型决定先翻哪边 |

## 项目进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| Proposal | ✅ 完成 | 已提交 |
| 数据集 | ✅ 完成 | LongMemEval oracle + s_cleaned 已下载 |
| 代码实现 | ✅ 完成 | 全部 6 种方法 + 3 个消融 + 评测 |
| 开发集验证 | ✅ 完成 | 50 条，确认流程跑通 |
| 正式实验 | ✅ 完成 | 150 条 × 3 seeds，oracle 数据 |
| S-Cleaned 实验 | ✅ 完成 | 100 条，大海捞针场景 |
| 消融实验 | ✅ 完成 | 去掉 relevance / diversity / routing |
| 案例分析 | ✅ 完成 | 80 条，成功/失败案例 |
| 改进实验 | ✅ 完成 | retrieval top-k、s_cleaned 调参、题型分析、LTM 质量分析 |
| LLM Answer Generation | ✅ 已支持 | 默认 dry-run 生成 prompts；有 API key 时可调用 OpenAI Responses API |
| 报告 v2 | ✅ 完成 | NeurIPS 格式，加入改进实验 |
| Presentation | ⬜ 待做 | 5/25–6/1 |
| 报告终稿 | ⬜ 待做 | 6/7 截止 |

## 关键实验结果

### Oracle 数据集（150 条 × 3 seeds）

| Method | Hit@6 | Recall@6 | Contains |
|--------|-------|----------|----------|
| STM-only | 0.642 | 0.478 | 0.342 |
| LTM-only | 0.713 | 0.549 | 0.309 |
| Flat Memory | 0.782 | 0.623 | 0.298 |
| Naive STM+LTM | 0.776 | 0.614 | 0.329 |
| Freq Promotion | **0.804** | **0.641** | 0.327 |
| SMC-QA (Ours) | 0.778 | 0.612 | **0.333** |

### 核心发现

1. **分层记忆有效**：只看最近消息（STM-only）最差，加了长期记忆后全部变好
2. **Promotion 有效**：有选择地提升记忆（Freq / SMC-QA）比简单合并好
3. **Diversity 最关键**：消融实验显示，"去重"（diversity）是提升最大的因子

### 改进实验结果

新增代码把补充实验正式纳入仓库：

| Experiment | Best / Main Result | 结论 |
|------------|--------------------|------|
| Promotion-only tuning | `baseline_thr0.40_top3_auto`: Hit@6 0.776, Recall@6 0.614, Contains 0.336 | 只调 promotion 对 Contains 略有帮助，但不能稳定提升 Hit@6 / Recall@6 |
| Retrieval top-k tuning | `baseline_thr0.40_top3_auto_stm8_ltm8`: Hit@6 0.793, Recall@6 0.631, Contains 0.329 | 把 STM/LTM retrieval depth 从 4 扩到 8 可以提升 retrieval coverage |
| S-Cleaned tuning | Tuned SMC-QA: Hit@6 0.510, Recall@6 0.360, Contains 0.270 | 在 100 条 s_cleaned sample 上超过 Flat Memory / Freq Promotion / Base SMC-QA |
| Question type analysis | temporal-reasoning: Hit@6 0.769 → 0.821 | retrieval tuning 对 temporal-reasoning 和 single-session-assistant 最明显 |
| LTM quality analysis | Base SMC-QA LTM recall 更高，tuned retrieval 最终指标更好 | 改进主要来自更宽的 retrieval depth，而不是 LTM 本身质量提升 |

## 项目结构

```
ml_project/
├── src/                        # 核心代码
│   ├── config.py               # 所有超参数
│   ├── data_loader.py          # 数据加载和抽样
│   ├── chunking.py             # 文本切块
│   ├── embedder.py             # 向量化 (MiniLM-L6-v2)
│   ├── memory.py               # STM / LTM / FlatMemory
│   ├── promotion.py            # 记忆提升策略
│   ├── router.py               # 查询路由
│   ├── evaluate.py             # 评测指标
│   ├── pipeline.py             # 实验管线（串联所有模块）
│   └── improvement.py          # 改进实验、调参、补充分析、answer generation 工具
├── run_experiments.py           # 一键跑全套实验
├── run_s_cleaned.py             # 大海捞针实验
├── run_case_analysis.py         # 案例分析
├── run_improvements.py          # 改进实验统一入口
├── data/                        # 数据文件（不上传到 git）
├── results/                     # 实验结果 JSON
├── Note/                        # baseline run 记录与结果
├── Improvement/                 # 改进探索记录；保留原始实验日志，不作为主入口
├── report/                      # LaTeX 报告
│   ├── report.tex               # base report
│   ├── report_v2.tex
│   └── neurips_2026.sty
└── guidelines/                  # 课程要求和设计文档
```

## 快速开始

```bash
# 1. 安装依赖
pip install sentence-transformers tqdm numpy

# 2. 下载数据（放到 data/ 目录下）
cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ..

# 3. 跑全套实验（dev + test×3 + ablation，约 10 分钟）
python run_experiments.py all

# 4. 跑大海捞针实验（约 5 分钟）
python run_s_cleaned.py

# 5. 案例分析
python run_case_analysis.py

# 6. 跑改进实验入口
python run_improvements.py --help
```

## 改进实验运行方式

所有正式改进实验入口都在 `run_improvements.py`。输出默认保存到 `results/`。

```bash
# Promotion-only 参数搜索
python run_improvements.py promotion

# Retrieval top-k 参数搜索
python run_improvements.py retrieval

# 题型 breakdown + LTM quality + case studies
python run_improvements.py supplementary

# S-Cleaned 专项调参
python run_improvements.py s-cleaned

# S-Cleaned 默认验证 tuning 排名前 5 的配置，也可以手动指定
python run_improvements.py s-cleaned --s-cleaned-top-configs 5

# 生成 answer generation prompts，不调用 API
python run_improvements.py answer-generation --n 20

# 如果需要实际调用 OpenAI Responses API
pip install openai
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.2"
python run_improvements.py answer-generation --n 20 --run-api
```

也可以一键运行除 LLM API 外的改进实验：

```bash
python run_improvements.py all
```

## 还需要做的事

- [ ] 编译报告（上传 `report/report_v2.tex` + `neurips_2026.sty` 到 Overleaf）
- [ ] 做 Presentation slides（5/25–6/1）
- [x] 接 LLM API 做 answer generation 实验入口（默认 dry-run）
- [x] 在 s_cleaned 上调优 promotion / retrieval 参数
- [ ] 报告终稿润色（6/7 截止）

## 指标说明

- **Hit@6**：翻出来的 6 条里有没有包含正确答案的段落（命中率）
- **Recall@6**：正确答案可能在好几段里，6 条覆盖了多少（覆盖率）
- **Contains**：翻出来的前 3 条拼在一起是否包含标准答案文本

## 技术细节

- 向量模型：`sentence-transformers/all-MiniLM-L6-v2`（384 维）
- 切块：120 词 / 30 词 overlap
- STM 容量：40 chunks（FIFO）
- LTM 容量：120 chunks（promotion 进入，去重阈值 0.85）
- Promotion 公式：`0.4 × relevance + 0.4 × reuse + 0.2 × diversity`
- Retrieval-tuned SMC-QA：`threshold=0.40`, `promotion_top_k=3`, `STM/LTM top_k=8`
- S-Cleaned tuned SMC-QA：`relevance=0.5`, `reuse=0.3`, `diversity=0.2`, `threshold=0.20`, `STM/LTM top_k=8`
- 数据集：LongMemEval（500 条，6 种题型）
