# SMC-QA: Selective Memory Consolidation with Query-Aware Retrieval for Long-Context QA

## 1. Project Overview

### 1.1 Problem Setting

本项目关注的是 **long-context question answering**。在这类任务中，系统需要面对长时间跨度、多轮交互或长文档场景中的历史信息，并从中找出真正有用的 evidence 来回答当前问题。

与传统短文本 QA 不同，long-context QA 的主要难点不仅是“理解问题”，还包括“如何管理记忆”。如果系统不能有效组织历史信息，就容易出现以下问题：

- 重要信息被淹没在大量无关内容中
- 重复信息不断累积，导致 retrieval noise 增加
- 早期关键信息难以在后续阶段再次被利用
- 新信息和旧信息冲突时，系统难以判断应优先依赖哪一部分

因此，本项目不打算实现一个通用 Agent，也不打算训练新的 LLM，而是聚焦于一个更清晰、更可控的问题：

> 在 long-context QA 中，如何通过层次化记忆、选择性记忆巩固和 query-aware retrieval，提高 evidence organization 与 evidence utilization 的效率？

### 1.2 Application Definition

从 application 角度看，SMC-QA 是一个 **Memory-Augmented Long-Context QA System**。它的角色可以概括为：

> 一个会管理历史信息、并基于记忆回答问题的 QA system。

它不是一个通用聊天机器人，也不是完整的多智能体系统，而是一个 **memory-aware evidence selector + QA wrapper**。

### 1.3 Inputs and Outputs

系统的输入有两类：

1. **Context Stream**
   一串按时间顺序到来的文本片段，可以是 document chunks、dialogue turns 或 benchmark 中的历史 context pieces。
2. **Query**
   当前系统需要回答的问题。

系统的输出是：

- **Answer**
- 可选的 `supporting evidence`

因此，这个项目最简洁的输入输出定义是：

- 输入 = `Context Stream + Query`
- 输出 = `Answer`

### 1.4 Core Motivation

本项目的核心动机是：**long-context QA 的性能瓶颈，往往不只在模型推理能力本身，而在于 memory 的写入、保留和读取机制。**

如果把所有历史信息都放进一个扁平的 memory pool，系统会面临以下局限：

- retrieval space 持续增大
- irrelevant / repeated / outdated information 不断干扰答案生成
- 近期信息与远期信息无法被区别对待
- 所有历史信息都被同样对待，缺乏长期保留机制

因此，我们希望通过两个核心设计来替代 flat memory：

- 用 **Short-Term Memory (STM)** 与 **Long-Term Memory (LTM)** 区分近期信息与沉淀信息
- 用显式的 **writing policy** 与 **reading policy** 控制 memory 的更新与访问

### 1.5 Research Hypothesis

本项目的核心假设是：

> 相比 flat memory，层次化记忆、选择性记忆巩固和 query-aware retrieval 可以在不显著增加系统复杂度的前提下，提高 evidence retrieval 质量，并进一步改善 long-context QA 的表现。

因此，SMC-QA 的目标不是追求一个更复杂的系统，而是在**可控复杂度**下验证 memory `organization / writing / reading` 三个层面的设计是否能带来稳定收益。

## 2. Contributions

本项目的贡献收敛为三个彼此清晰分工的部分，分别对应 memory 的 `organization / writing / reading`。

### 2.1 Hierarchical Memory Framework

我们构建一个适用于 long-context QA 的**轻量、可复现的层次化记忆实验框架**，将历史信息组织为 **STM** 和 **LTM**，并提供统一的 evidence access interface。

这一部分解决的问题是：

> 信息应该如何存储？

### 2.2 Selective Memory Consolidation

我们设计一种 **Selective Memory Consolidation** 策略，用于决定哪些 short-term information 值得被长期保留。与简单地把所有 STM 内容写入 LTM 不同，我们根据 **Relevance**、**Reuse** 和 **Diversity** 来决定 promotion。

这一部分解决的问题是：

> 什么值得长期保留？

### 2.3 Query-Aware Retrieval

我们设计一种 **Query-Aware Retrieval** 机制，使系统能够根据 query 特征动态访问 STM、LTM 或二者融合，并从不同 memory levels 中提取 evidence。

这一部分解决的问题是：

> 面对当前问题，应该如何读取 memory？

说明：baseline comparison 很重要，但它属于 `Experiment Design` 中的验证部分，而不是单独的方法贡献点。

## 3. System Workflow

从整体上看，SMC-QA 的 workflow 可以固定为五个阶段。

| Stage | 名称                           | 核心作用                                       | 主要操作                                                         | 输入                                       | 输出                                 |
| ----- | ------------------------------ | ---------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------ | ------------------------------------ |
| 1     | Context Ingestion              | 将原始历史信息转换成可写入 memory 的单元       | chunk / turn 切分、embedding 编码、附加 metadata、写入 STM       | raw context chunks / dialogue turns        | STM memory items                     |
| 2     | Short-Term Memory Maintenance  | 保留 recent information 并维护短期状态         | 维护 item-level metadata、保留 recent chunks、准备 consolidation | STM memory items                           | updated STM memory pool              |
| 3     | Selective Memory Consolidation | 决定哪些 short-term items 值得长期保留         | 计算 `Relevance / Reuse / Diversity`、执行 promotion、更新 LTM | STM candidates, query history, current LTM | promoted items, updated LTM          |
| 4     | Query-Aware Retrieval          | 根据 query 从合适的 memory level 中找 evidence | routing、top-k retrieval、merge、deduplication、rerank           | query, STM, LTM                            | final evidence set                   |
| 5     | Answer Generation              | 基于 query 和 retrieved evidence 输出答案      | evidence 拼接、输入 QA module / LLM、生成 answer                 | query, retrieved evidence                  | answer, optional supporting evidence |

### Workflow Summary

整个系统可以压缩成下面这条主线：

`Context Stream -> STM --(Selective Consolidation)--> LTM ; Query -> Query-Aware Retrieval over STM/LTM -> Answer`

从项目边界来看，我们的研究重点是 `memory organization + consolidation + retrieval`；`Answer Generation` 主要作为下游验证模块，不是本项目的核心创新点。

## 4. Method Design

### 4.1 Memory Representation

为控制实现复杂度，本项目第一版统一采用 **chunk-level memory unit**。也就是说，原始长文本或对话历史会先被切分成固定长度或滑动窗口的文本块，每个文本块作为一个基本 memory item。

| 设计项         | 选型                      | 原因                                         |
| -------------- | ------------------------- | -------------------------------------------- |
| memory unit    | chunk-level               | 比 sentence-level 更稳定，不容易过碎         |
| chunking strategy | fixed-size chunk with fixed overlap | 便于实现与复现，避免 chunking 策略本身成为额外变量 |
| retrieval 表示 | dense embedding           | 更容易与 dense retrieval 对接                |
| item 粒度      | 统一 chunk 粒度           | 更适合统一的 promotion 和 deduplication 逻辑 |
| 适配场景       | long-context QA benchmark | 比 turn-level 更通用                         |

第一版 `memory item` 至少包含以下字段：

| 字段            | 含义                |
| --------------- | ------------------- |
| text            | 原始文本内容        |
| order_index     | 写入顺序或时间索引  |
| embedding       | 向量表示            |
| retrieval_count | 被检索次数          |
| memory_level    | 当前属于 STM 或 LTM |

如果后续有余力，再增加 `source_id`、`topic_tag`、`summary` 或其他 metadata。

需要强调的是，**chunking strategy 在本项目中固定，不作为研究变量**。第一版实现只采用统一的 fixed-size chunking，不比较不同切分策略。

### 4.2 Hierarchical Memory Structure

| Memory Level | 角色                         | 主要特点                                                          | 更适合的问题类型             |
| ------------ | ---------------------------- | ----------------------------------------------------------------- | ---------------------------- |
| STM          | 保存最近写入、尚未稳定的信息 | recent information、多细节、对近期 query 更敏感、可能包含更多噪声 | 依赖近期上下文的问题         |
| LTM          | 保存跨时间仍然有价值的信息   | 更稳定、信息密度更高、适合长期复用、必须控制 redundancy           | 依赖远程历史或跨轮信息的问题 |

#### Initial Parameters

为了让方案真正可落地，第一版系统采用明确参数设定：

| 参数                 | 含义               | 第一版设定                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------ |
| `N`                | STM 容量           | STM 保留最近 `N` 个 chunks                                 |
| M                    | consolidation 频率 | 每处理 `M` 个 queries 或每个固定窗口执行一次 consolidation |
| K                    | 每轮 LTM 晋升数量  | 每轮只接收 `top-K` promoted items                          |
| similarity threshold | LTM 去重阈值       | 对 promoted items 做 similarity-based deduplication          |

具体超参数 `N`、`M`、`K` 通过小规模 validation 确定，但系统结构固定为：

`bounded STM + periodic consolidation + bounded LTM`

在第一版实现中，可采用以下默认参考范围作为起点，而不是把这些参数完全留空：

| 参数 | 默认参考范围 |
| --- | --- |
| `N` | recent `20-50` chunks |
| `M` | 每 `5-10` 个 queries 做一次 consolidation |
| `K` | 每轮 promotion 取 `top-3` 或 `top-5` |

这些默认值不是研究重点，只用于帮助系统尽快进入可运行状态。

### 4.3 Selective Memory Consolidation

这是 SMC-QA 的核心写入模块。我们不让所有 STM 内容自动进入 LTM，而是只保留真正值得长期存储的内容。

对于每个 candidate item，我们定义：

`Score = alpha * Relevance + beta * Reuse + gamma * Diversity`

| 因素      | 含义                                     | 第一版实现方式                                                                  | 设计目的                               |
| --------- | ---------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------- |
| Relevance | item 与近期 query context 的平均关联程度 | 最近 `W` 个 queries 与该 item 的平均相似度，或近期窗口内的平均 matching score | 避免 promotion 只受单个当前 query 影响 |
| Reuse     | item 在近期是否被反复检索或使用          | 最近 `W` 个 queries 中的命中次数，或归一化 retrieval count                    | 反映“近期持续价值”而不是历史累计优势 |
| Diversity | item 是否为 LTM 提供新信息               | 计算 candidate 与当前 LTM 最近邻条目的最大相似度；超过阈值则降分或不晋升        | 防止 LTM 被高相似重复内容填满          |

在第一版实现中，`alpha / beta / gamma` 默认采用 **equal weights**。权重不是本项目的核心研究对象，只做小范围调参，不做系统性权重搜索。

同时，如果 query-history-based scoring 的实现复杂度过高，第一版允许退化为基于当前窗口 retrieval statistics 的近似版本，以降低工程风险。

#### Promotion Rule

为了保证系统稳定和可控，第一版实现固定采用：

**Top-k promotion + similarity-based deduplication**

| 步骤 | 操作                                | 作用                    |
| ---- | ----------------------------------- | ----------------------- |
| 1    | 计算 STM candidate items 的综合得分 | 得到 promotion priority |
| 2    | 选取得分最高的 top-k items          | 控制每轮晋升数量        |
| 3    | 与现有 LTM 做相似度去重             | 防止重复内容进入 LTM    |
| 4    | 将最终保留条目写入 LTM              | 更新长期记忆            |

这种设计的优点是：

- 容易控制 LTM 大小
- 比 threshold-based promotion 更稳定
- 更适合课程项目中的对比实验

### 4.4 Query-Aware Retrieval

这是 SMC-QA 的核心读取模块。系统不会固定只从某一层 memory 检索，而是根据 query 特征决定 retrieval path。

#### Routing Principle

不同类型的问题依赖的 memory range 不同，因此 retrieval path 也应不同。

| Query Type      | 判断依据                                        | Retrieval Path | 说明                     |
| --------------- | ----------------------------------------------- | -------------- | ------------------------ |
| Recent-focused  | query 与 recent context 更相关                  | `STM-first`  | 优先利用近期细粒度信息   |
| History-focused | query 含明显历史 / 时间指示词，或更依赖远程事实 | `LTM-first`  | 优先利用长期沉淀信息     |
| Mixed           | 同时涉及近期状态与历史事实                      | `Hybrid`     | 同时从 STM 与 LTM 取证据 |

#### First-Version Router

为避免复杂化，第一版 router 固定采用 **rule-based routing**，**不训练额外 query classifier，也不把 query classification 作为单独研究问题**。核心规则如下：

| Rule   | 条件                                                            | 路由结果      |
| ------ | --------------------------------------------------------------- | ------------- |
| Rule 1 | query 中包含“之前”“最早”“曾经”“后来”等历史 / 时间指示词 | `LTM-first` |
| Rule 2 | query 与 recent context 的平均相似度显著高于 LTM candidates     | `STM-first` |
| Rule 3 | 不满足以上条件                                                  | `Hybrid`    |

#### Retrieval Pipeline

为控制复杂度，retrieval 后处理固定为：

`retrieve -> merge -> deduplicate -> rerank`

| 步骤 | 操作            | 作用                                      |
| ---- | --------------- | ----------------------------------------- |
| 1    | `retrieve`    | 从 STM 和 / 或 LTM 中取 top-k candidates  |
| 2    | `merge`       | 合并多路候选 evidence                     |
| 3    | `deduplicate` | 移除高相似度重复 evidence                 |
| 4    | `rerank`      | 按 query-evidence matching score 重新排序 |

第一版系统不再额外引入复杂的 source weighting 或更重的 fusion strategy。

对于所有需要双层联合检索的设置，这条 pipeline 保持一致，不作为单独研究变量展开比较。

### 4.5 Relation to Baselines

SMC-QA 相比基线方法的主要差异在于：

- 相比 `Flat Memory`，我们显式区分 STM 与 LTM
- 相比 `STM-only` 或 `LTM-only`，我们允许双层协同
- 相比 `Naive STM+LTM`，我们加入 query-aware routing
- 相比 `Frequency-based Promotion`，我们同时考虑 Relevance、Reuse 和 Diversity
- 相比 `Summary-based Promotion`，我们不依赖额外 summary module，结构更轻

## 5. Experiment Design

### 5.1 Research Questions

实验不只是为了证明 “ours 最好”，而是为了回答下面这些问题：

| 编号 | Research Question                                                | 对应验证内容                                                              |
| ---- | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| RQ1  | 层次化记忆是否优于 `flat memory`？                             | 比较 `Flat Memory`、`STM-only`、`LTM-only`、`STM+LTM Naive Merge` |
| RQ2  | `Selective Memory Consolidation` 是否优于简单 promotion 规则？ | 比较 `Frequency-based`、`Summary-based` 与 `SMC-QA`                 |
| RQ3  | `Query-Aware Retrieval` 是否能提高 evidence selection 质量？   | 比较固定双层检索与 query-aware routing                                    |
| RQ4  | `STM` 和 `LTM` 在不同 query 类型下分别起到什么作用？         | 做 query type analysis 与错误案例分析                                     |

### 5.2 Dataset

主数据集计划设为 **LongMemEval**。它能够较好覆盖 long-term memory、multi-session reasoning 和 information update 等典型场景，适合作为 SMC-QA 的主要 benchmark。

为避免数据集选择反复摇摆，第一方案固定为 **LongMemEval**；如果工程接入成本过高，fallback 方案优先是**只跑 LongMemEval 的部分任务子集**，而不是立即更换整套 benchmark。

| 数据集角色 | 数据集 / 方案 | 作用 |
| ---------- | ------------- | ---- |
| 主 benchmark | `LongMemEval` | 作为主要 benchmark，验证 long-term memory 与 cross-session reasoning 场景 |
| Fallback 方案 | `LongMemEval` 子任务 / 子集 | 当全量接入成本过高时，优先保证主 benchmark 的可运行原型 |
| Optional secondary benchmark | `TBD` | 仅在时间允许时补充，不作为必须项 |

### 5.3 Baselines

为了让对比结构清晰，baselines 分为三组：

| 类别                         | 方法                      | 说明                                                                                        |
| ---------------------------- | ------------------------- | ------------------------------------------------------------------------------------------- |
| Retrieval Structure Baseline | Flat Memory               | 所有 memory items 放在一个统一池中检索                                                      |
| Retrieval Structure Baseline | STM-only Retrieval        | 只从 STM 中检索                                                                             |
| Retrieval Structure Baseline | LTM-only Retrieval        | 只从 LTM 中检索                                                                             |
| Retrieval Structure Baseline | STM+LTM Naive Merge       | 从两层检索后直接拼接，不做 query-aware routing                                              |
| Promotion Baseline           | Frequency-based Promotion | 根据被检索频率决定是否晋升                                                                  |
| Promotion Baseline           | Summary-based Promotion   | 先做 summary，再将 STM 信息写入 LTM；若实现成本过高，可降级为 optional baseline            |
| Ours                         | SMC-QA                    | 使用 `Relevance + Reuse + Diversity` 做 selective promotion，并结合 query-aware retrieval |

### 5.4 Metrics

我们将指标分为主指标和分析指标。

| 类别                    | Metric                         | 作用                               |
| ----------------------- | ------------------------------ | ---------------------------------- |
| Primary Metric          | QA accuracy / benchmark 主指标 | 衡量最终回答质量                   |
| Primary Metric          | `Recall@k`                   | 衡量 evidence retrieval 的覆盖能力 |
| Primary Metric          | `Hit@k`                      | 衡量 top-k 检索命中率              |
| Optional Primary Metric | `Exact Match`                | 数据集支持时补充                   |
| Optional Primary Metric | `F1`                         | 数据集支持时补充                   |
| Analysis Metric         | `LTM size`                   | 分析长期记忆规模控制效果           |
| Analysis Metric         | `redundancy rate`            | 分析重复 memory 的比例             |
| Analysis Metric         | `retrieval latency`          | 分析检索开销                       |

在最终实验呈现中，**主结果表优先报告**：

- `QA accuracy` / benchmark 主指标
- `Recall@k`
- `Hit@k`

而 `Exact Match`、`F1`、`LTM size`、`redundancy rate`、`retrieval latency` 作为附加分析或补充结果呈现。

### 5.5 Main Experiments

主实验比较所有 baselines 与 SMC-QA 在同一数据集上的表现，重点观察：

- QA performance
- retrieval quality

主实验的目标是比较不同 memory designs 在统一设置下的行为差异，而不是在文档中预设过多结论。

### 5.6 Ablation Studies

| Ablation 类型                | 设置                           | 验证目标                           |
| ---------------------------- | ------------------------------ | ---------------------------------- |
| Promotion Factor Ablation    | remove `Relevance`           | 验证 relevance 对 promotion 的贡献 |
| Promotion Factor Ablation    | remove `Reuse`               | 验证 reuse 对 promotion 的贡献     |
| Promotion Factor Ablation    | remove `Diversity`           | 验证 diversity 对 promotion 的贡献 |
| Retrieval Mechanism Ablation | remove `query-aware routing` | 验证 routing 是否真正带来收益      |
| Retrieval Mechanism Ablation | remove `deduplication`       | 验证 evidence 去重的重要性         |
| Retrieval Mechanism Ablation | remove `rerank`              | 验证后排序的重要性                 |

若时间紧张，ablation 的优先级建议为：

1. remove `query-aware routing`
2. remove `Relevance`
3. remove `Reuse`
4. remove `Diversity`
5. remove `deduplication` / `rerank`

### 5.7 Analysis Studies

| 优先级   | Analysis Study       | 关注点                                                                           |
| -------- | -------------------- | -------------------------------------------------------------------------------- |
| Required | Query Type Analysis  | 分析 `STM / LTM` 在近期依赖型、历史依赖型和混合型 query 下的使用比例与性能差异 |
| Required | Error Case Analysis  | 分析 promotion 失败、routing 错误、retrieval 成功但 answer generation 失败等案例 |
| Optional | LTM Quality Analysis | 分析 LTM 中的重复、低价值条目累积与关键信息保留情况                              |

## 6. Code Design

### 6.1 Design Principle

代码实现遵循一个原则：**保持 pipeline 清晰，避免实现成复杂的 agent system。**

整个系统可以抽象成：

`answer = system.run(context_stream, query)`

其内部主线为：

- `write(chunk)`
- `consolidate(query_history)`
- `retrieve(query)`
- `answer(query, evidence)`

为了突出 memory design 的作用，**Answer Generation module 在所有实验中尽量保持固定，不作为单独研究变量展开比较。**

### 6.2 Core Modules

| 模块         | 主要职责                                                      | 关键输入                                | 关键输出                              |
| ------------ | ------------------------------------------------------------- | --------------------------------------- | ------------------------------------- |
| data/        | 数据读取、chunk 切分、benchmark 适配                          | raw dataset                             | standardized context stream           |
| memory/      | 定义 `MemoryItem`，管理 `STM / LTM`                       | chunks, promoted items                  | STM/LTM memory pools                  |
| promotion/   | 计算 `Relevance / Reuse / Diversity`，执行 promotion        | STM items, query history, LTM           | promoted items, promotion scores      |
| retrieval/   | embedding、similarity、top-k retrieval、deduplication、rerank | query, memory items                     | ranked evidence                       |
| router/      | 决定 `STM / LTM / Hybrid` 路径                              | query, recent context, similarity stats | retrieval route                       |
| qa/          | 将 query 和 evidence 输入下游 QA module                       | query, ranked evidence                  | final answer                          |
| experiments/ | 编排 baselines、ablations、evaluation                         | configs, modules                        | result tables, logs, analysis outputs |

实现时需要保持两个边界清晰：

- `router/` 只负责决定 retrieval path，不直接执行 retrieval
- `retrieval/` 只负责 retrieval、merge、deduplication 与 rerank，不负责 routing decision

同时，`qa/` 保持极简，只负责 final answer generation，不扩展为 reasoning controller。

### 6.3 Minimal Viable Prototype

为降低风险，实现顺序采用分阶段推进。

| 阶段  | 范围                                                                                                    | 目标                                                | 验收标准 |
| ----- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------- |
| MVP-1 | chunk-level memory、`STM / LTM`、`Flat Memory`、`STM-only`、`LTM-only`、`STM+LTM Naive Merge` | 先跑通 hierarchical memory framework 与基础检索闭环 | 至少能跑通 4 个 retrieval structure baselines |
| MVP-2 | `Frequency-based Promotion`、`Summary-based Promotion`、`Selective Memory Consolidation`          | 跑通 promotion strategy 并与 baseline 比较          | 至少能比较 3 种 promotion strategy |
| MVP-3 | rule-based router、`STM-first / LTM-first / Hybrid retrieval`                                         | 跑通 query-aware retrieval 与完整 SMC-QA pipeline   | 至少能完成 SMC-QA end-to-end 运行 |

### 6.4 Suggested Project Structure

```text
smc_qa/
├── data/
├── configs/
├── src/
│   ├── memory.py
│   ├── retriever.py
│   ├── consolidator.py
│   ├── router.py
│   ├── fusion.py
│   ├── qa_module.py
│   ├── manager.py
│   └── evaluate.py
├── scripts/
│   ├── run_flat.py
│   ├── run_stm_only.py
│   ├── run_ltm_only.py
│   ├── run_naive_merge.py
│   ├── run_freq_promotion.py
│   ├── run_summary_promotion.py
│   └── run_smc_qa.py
└── results/
```

### 6.5 Recommended Implementation Order

| 顺序 | 实现内容                                    | 目的                        |
| ---- | ------------------------------------------- | --------------------------- |
| 1    | 数据处理与 chunk 切分                       | 搭好 context ingestion 输入 |
| 2    | `MemoryItem`、STM、LTM、`MemoryManager` | 跑通 memory framework       |
| 3    | baseline retrieval pipeline                 | 跑通基础 evidence retrieval |
| 4    | retrieval structure baselines               | 建立第一组对照实验          |
| 5    | promotion baselines                         | 建立第二组对照实验          |
| 6    | selective consolidation                     | 实现核心 writing policy     |
| 7    | query-aware router                          | 实现核心 reading policy     |
| 8    | analysis scripts                            | 产出消融与分析结果          |

## 7. Milestones

| Milestone                        | 主要任务                                                                        | 交付物                                        |
| -------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------- |
| M1: Basic Framework              | 完成数据预处理与 chunk pipeline；完成 STM / LTM 基本结构                        | 可运行的 context ingestion + memory framework |
| M2: Retrieval Baselines          | 完成 Flat / STM-only / LTM-only / Naive Merge                                   | 第一版 baseline 结果表                        |
| M3: Promotion Baselines and Ours | 完成 `Frequency-based`、`Summary-based`、`Selective Memory Consolidation` | promotion 对比结果                            |
| M4: Query-Aware Retrieval        | 完成 rule-based router；完成 `retrieve -> merge -> deduplicate -> rerank`     | 端到端 SMC-QA pipeline                        |
| M5: Experiments and Analysis     | 跑通主实验、消融实验、query type analysis、error case analysis                  | 主结果表、消融表、分析图                      |
| M6: Report and Presentation      | 整理 final report；准备 presentation slides；清理代码与配置                     | 可提交报告、展示材料和代码包                  |

### Suggested Timeline

#### Ideal Plan

| 周次    | 计划内容                                               |
| ------- | ------------------------------------------------------ |
| 第 1 周 | 完成 memory framework 和 retrieval structure baselines |
| 第 2 周 | 完成 promotion baselines                               |
| 第 3 周 | 完成 selective consolidation                           |
| 第 4 周 | 完成 query-aware retrieval 和 end-to-end pipeline      |
| 第 5 周 | 完成 experiments、ablations 和 analysis                |

#### Fallback Plan

若项目中期遇到时间压力，优先做以下收缩：

- 保留 `LongMemEval`，但只跑其核心子任务或子集
- 保留 `Query Type Analysis` 和 `Error Case Analysis`，将 `LTM Quality Analysis` 降为 optional
- 保留 `Frequency-based Promotion`，将 `Summary-based Promotion` 降为 optional baseline
- 保留核心 ablations，降低 `deduplication / rerank` 相关消融优先级

## 8. Scope and Non-goals

为了控制项目边界，本项目明确以下内容**不作为主要研究目标**：

- 不训练新的 LLM
- 不把 `Answer Generation` 作为主要研究变量
- 不研究 chunking strategy 本身
- 不训练复杂的 query classifier
- 不构建多智能体 pipeline 或通用 agent system

## 9. One-Sentence Summary

SMC-QA 是一个面向 long-context QA 的轻量 memory-augmented system，它通过 **Hierarchical Memory Framework** 组织历史信息，通过 **Selective Memory Consolidation** 控制长期记忆质量，再通过 **Query-Aware Retrieval** 从不同 memory levels 中提取 evidence，最终支持 answer generation。
