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
| 报告初稿 | ✅ 完成 | NeurIPS 格式，~7 页 |
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

## 项目结构

```
ml_prejects/
├── src/                        # 核心代码
│   ├── config.py               # 所有超参数
│   ├── data_loader.py          # 数据加载和抽样
│   ├── chunking.py             # 文本切块
│   ├── embedder.py             # 向量化 (MiniLM-L6-v2)
│   ├── memory.py               # STM / LTM / FlatMemory
│   ├── promotion.py            # 记忆提升策略
│   ├── router.py               # 查询路由
│   ├── evaluate.py             # 评测指标
│   └── pipeline.py             # 实验管线（串联所有模块）
├── run_experiments.py           # 一键跑全套实验
├── run_s_cleaned.py             # 大海捞针实验
├── run_case_analysis.py         # 案例分析
├── data/                        # 数据文件（不上传到 git）
├── results/                     # 实验结果 JSON
├── report/                      # LaTeX 报告
│   ├── report.tex
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
```

## 还需要做的事

- [ ] 编译报告（上传 `report/report.tex` + `neurips_2026.sty` 到 Overleaf）
- [ ] 做 Presentation slides（5/25–6/1）
- [ ] 可选：接 LLM API 做 answer generation 实验
- [ ] 可选：在 s_cleaned 上调优 promotion 参数
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
- 数据集：LongMemEval（500 条，6 种题型）
