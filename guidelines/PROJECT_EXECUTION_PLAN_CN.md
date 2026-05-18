# SMC-QA 项目执行方案（中文可直接开工版）

## 1. 这份文档是干什么的

这不是“灵感草稿”，而是一份接下来可以直接照着执行的项目方案。

它要解决的问题是：

- 这个项目到底做什么
- 为什么这样做
- 每一步怎么做
- 参数怎么定
- 模型怎么选
- 实验怎么跑
- 结果怎么判
- 如果中途卡住，怎么降级而不换题

---

## 2. 课程要求到底要我们做成什么样

这门课里你们选的是：

## **Category 2: Applied / Engineering**

老师真正想看到的是一个：

- 用真实数据
- 解决明确问题
- 能运行
- 有比较实验
- 能解释结果

的机器学习工程项目。

所以我们最后要交的，不是“一个很会讲概念的设计稿”，而是：

1. 明确的问题定义
2. 一个真实数据集
3. 至少 3 种以上可运行方法比较
4. 你们自己的方法
5. 主结果表
6. 一些分析和案例

这就是我们这份方案的所有设计原则。

---

## 3. 现有 proposal 和设计稿的问题

现有的：

- [MAT8034_Final_Project_Proposal.md](/home/speediance/projects/ml_prejects/guidelines/MAT8034_Final_Project_Proposal.md)
- [SMC_QA.md](/home/speediance/projects/ml_prejects/guidelines/SMC_QA.md)

它们的优点是：

- 方向已经定了
- 题目能讲通
- 故事线比较完整

但它们现在的问题也很明显：

- 还偏“空中楼阁”
- 讲了很多“想做什么”，但没有把“怎么做”具体到能立刻开工
- 没把实验细节定死
- 没把调参方法定死
- 没把模型和评分方式定死

所以这份文档的任务就是把它们压成“可执行版本”。

---

## 4. 我们最后到底要做什么

我们最后要做的是一个：

## **会整理历史笔记、再拿笔记回答问题的系统**

系统流程用最白话的话说就是：

1. 输入一段很长的历史内容
2. 把它切成很多小段
3. 最近的小段先放进“短期记忆”
4. 从短期记忆里挑出值得保留的内容，放进“长期记忆”
5. 问题来了以后，决定先查短期记忆、长期记忆，还是两边都查
6. 把查到的几条证据交给同一个回答模块，生成最终答案

你们的方法叫 `SMC-QA`，核心不是“训练一个新大模型”，而是：

- **怎么存**
- **怎么挑**
- **怎么查**

---

## 5. 和已提交 proposal 的关系

这份方案不是换题，只是把 proposal 做成一个可交付的工程版。

### 5.1 保留的主线

- long-context QA
- STM / LTM 两层记忆
- selective memory consolidation
- query-aware retrieval
- baseline comparison

### 5.2 主动收缩的部分

- 不做复杂 summary-based promotion 第一版
- 不训练 router，只做规则版
- 不一开始跑完整个 benchmark
- 不把 answer generation 当研究重点

### 5.3 最终对外说法

如果最后报告里要解释，可以这样写：

> Proposal 中提出了完整研究方向；在正式实现阶段，我们保留核心问题与方法主线，并对非核心模块做范围控制，以确保项目在课程时间内可复现、可比较、可交付。

---

## 6. 最终项目目标

### 6.1 核心问题

我们要验证的是：

> 在 long-context QA 中，把历史信息分成短期记忆和长期记忆，并只把有价值的信息写入长期记忆，是否比简单的统一检索更有效？

### 6.2 我们要证明的不是“最强”，而是这三件事

1. 分层记忆比 flat memory 更合理
2. 有选择地保留长期信息比简单堆积更好
3. 不同问题适合查不同层级记忆

---

## 7. 文献依据和我们自己的工程设定要分清楚

这里要明确区分两类东西。

### 7.1 来自文献和现有工作思路的部分

这些想法不是拍脑袋来的：

- 长期 / 分层记忆适合 long-horizon QA
- 不是所有历史内容都应该永久保留
- 相关性、复用性、去冗余是合理的记忆筛选维度
- 检索路径可以根据问题类型变化

和这些思路关系最密切的参考包括：

1. **LongMemEval**
   作用：数据和任务定义参考
2. **HiMem**
   作用：层次化长期记忆设计思路参考
3. **Reflective Memory Management (RMM)**
   作用：长期信息整理与多粒度记忆灵感来源
4. **LightMem**
   作用：轻量 memory module 拆分思路参考

### 7.2 由我们自己工程化定下来的部分

这些不是论文直接给死的标准答案，而是为了把系统做出来，我们必须定的第一版参数：

- `chunk size = 120`
- `overlap = 30`
- `STM capacity = 40`
- `LTM max size = 120`
- 每 `5` 个 query consolidation 一次
- `score = 0.4 * relevance + 0.4 * reuse + 0.2 * diversity`
- `score threshold = 0.55`
- `top-3 promotion`
- `similarity >= 0.85` 视为重复

所以未来报告里要写清楚：

> 我们参考已有工作中的核心思想，但将具体阈值、容量和权重作为工程超参数，在开发集上进行小规模调参确定。

---

## 8. 数据方案

### 8.1 主数据集

主数据集固定：

- `LongMemEval`

原因：

- 与 proposal 一致
- 专门和长期记忆 / 多轮历史有关
- 最容易解释为什么你们这个方法有意义

### 8.2 数据使用范围

第一版固定：

- 总样本数：`200`
- 开发调参集：`50`
- 正式实验集：`150`

如果后面非常顺利，再扩成：

- 开发调参集：`100`
- 正式实验集：`300`

### 8.3 这 200 条怎么抽

原则固定如下：

1. 先按 LongMemEval 原始任务结构读入全部样本
2. 如果有子任务标签，先只保留最贴近“历史记忆问答”的子任务
3. 从保留下来的样本里按固定随机种子抽样

第一版随机种子固定：

- `seed = 42`

### 8.4 50 条开发调参集再怎么拆

这 50 条不是“只看一眼”，而是要继续拆成：

- `30` 条：开发搜索集
- `20` 条：开发确认集

作用分别是：

- 30 条：试参数
- 20 条：防止你在 30 条上乱调到过拟合

### 8.5 如果 LongMemEval 接入成本过高

降级顺序固定：

1. 先只跑一个子任务
2. 再只跑固定数量样本
3. 最后才考虑换数据集

也就是说：

**优先缩数据范围，不优先换题。**

---

## 9. 数据切分方法

### 9.1 切分单位

第一版固定用：

- `chunk-level`

### 9.2 切分规则

默认规则：

- `chunk size = 120 words`
- `overlap = 30 words`

如果原始样本是对话型：

- 优先保留自然 turn
- 只有当单个 turn 超过 `160 words` 时再切块

### 9.3 为什么是这个数

选择逻辑：

- 小于 `100` words：太碎，容易切断完整信息
- 大于 `150` words：一块太杂，检索不稳
- `120 + 30 overlap` 是一个折中值

### 9.4 本项目不研究 chunking

必须锁死：

- 不比较 sentence-level
- 不比较多种 overlap 策略作为最终研究问题

最多只在开发调参阶段试一小组候选值，确定后就冻结。

---

## 10. 记忆结构

### 10.1 STM 是什么

STM 就是“最近的、还没整理过的记忆”。

它保留更多细节，但也更乱。

### 10.2 LTM 是什么

LTM 就是“整理后留下来的长期记忆”。

它不能无限增长，否则会变成第二个垃圾堆。

### 10.3 STM 固定规则

- `STM capacity = 40 chunks`

行为规则：

- 新 chunk 进入 STM
- 超过 40 个后，最旧的先移出 STM

### 10.4 LTM 固定规则

- `LTM max size = 120 chunks`

行为规则：

- 只有通过 promotion 的 chunk 才能进入 LTM
- 如果 LTM 超过 120，则删除“长期没被用到且高重复”的条目

---

## 11. 长期记忆挑选方法

这一步是我们项目的核心方法部分。

### 11.1 核心思路

不是所有短期记忆都进长期记忆。

每次只挑“最近确实有价值、而且不重复”的少数内容。

### 11.2 候选评分公式

第一版固定：

`score = 0.4 * relevance + 0.4 * reuse + 0.2 * diversity`

### 11.3 三个分数怎么计算

#### relevance

定义：

- 这条 chunk 和最近问题的相关程度

计算：

- 取最近 `5` 个 query
- 计算 chunk 与这 `5` 个 query embedding 相似度平均值
- 再归一化到 `0~1`

#### reuse

定义：

- 这条 chunk 最近是不是经常被取出来

计算：

- 统计最近 `5` 个 query 中它被命中的次数
- 除以理论最大命中次数，归一化到 `0~1`

#### diversity

定义：

- 它是不是在给 LTM 带来新信息，而不是重复旧信息

计算：

- 找到当前 LTM 里和它最像的一条
- 记这条最大相似度为 `max_sim`
- `diversity = 1 - max_sim`

### 11.4 promotion 规则

每次 consolidation 时：

1. 对 STM 候选算分
2. 取分数最高的 `top-3`
3. 只有 `score >= 0.55` 才允许进入 LTM
4. 若与现有 LTM 条目相似度 `>= 0.85`，则拒绝写入

### 11.5 这个流程的论文依据怎么写

可以这样写进报告：

> 我们参考了长期记忆工作中“相关性优先”“高频复用优先”“避免记忆冗余”的通用思想，构建了一个轻量 promotion score。相比复杂 memory writer，本方法不训练额外网络，重点强调课程项目中的可复现性与对比性。

---

## 12. 根据问题选检索路径的方法

### 12.1 为什么这里需要 Rule 1 / 2 / 3

因为 `query-aware retrieval` 不是一句空话，必须变成具体规则。

这些 rule 不是 baseline。

它们是：

- **Ours 方法内部的一个模块**

baseline 回答的是：

- 哪种方法更好？

rule 回答的是：

- 你们的方法内部，到底怎么决定先查哪种记忆？

### 12.2 Rule 1

如果 query 中有明显历史词，比如：

- 之前
- 最早
- 曾经
- 后来
- before
- earlier
- previously
- first

则：

- `LTM-first`

### 12.3 Rule 2

如果 query 和最近 `10` 个 STM chunks 的平均相似度

比它和 LTM 的平均相似度高至少 `0.05`

则：

- `STM-first`

### 12.4 Rule 3

其他情况：

- `Hybrid`

也就是两边都查。

### 12.5 最终取证据数

默认固定：

- `STM top-k = 4`
- `LTM top-k = 4`

如果单路检索：

- 只取该路 `top-4`

如果双路检索：

- 各取 `4`
- 合并去重
- 最终 rerank 后保留 `top-6`

---

## 13. Baseline 具体定义

这里不能只列名字，要把每个 baseline 的运行方式定死。

### 13.1 Baseline A: Flat Memory

规则：

- 所有 chunks 放进一个统一池
- 直接对 query 做检索
- 最终取 `top-6`

作用：

- 测试“不分层”的最简单方案

### 13.2 Baseline B: STM-only

规则：

- 只从 STM 检索
- 取 `top-4`
- 如果不足 4 条，就取现有全部

作用：

- 测试只依赖近期记忆

### 13.3 Baseline C: LTM-only

规则：

- 只从 LTM 检索
- 取 `top-4`

作用：

- 测试只依赖长期保留记忆

### 13.4 Baseline D: Naive STM+LTM

规则：

- STM 取 `top-4`
- LTM 取 `top-4`
- 直接合并、去重、重排
- 不判断 query 类型

作用：

- 测试“有两层，但不会智能分流”

### 13.5 Baseline E: Frequency-based Promotion

规则：

- 写入 LTM 时不看 relevance / diversity
- 只看最近被命中的次数
- 每轮 promotion 取 reuse 最高的 `top-3`

作用：

- 测试“只按被用次数写长期记忆”是否够用

### 13.6 Ours: SMC-QA

规则：

- STM + LTM
- `relevance + reuse + diversity`
- Rule 1 / Rule 2 / Rule 3 决定检索路径

作用：

- 完整验证你们的方法主张

### 13.7 为什么第一版不做 summary-based promotion

因为它会额外引入一个“先做总结，再写记忆”的模块。

这会新增：

- 总结质量问题
- 总结模型选择问题
- 总结是否丢信息的问题

工程成本明显提高。

所以第一版明确降级：

- **不做必选 baseline**
- 只在时间非常宽松时补成 optional baseline

---

## 14. 检索模型和开源资源怎么选

### 14.1 检索 embedding 首选

第一版首选：

- `sentence-transformers/all-MiniLM-L6-v2`

原因：

- 轻量
- 英文任务常用
- 本地容易跑
- 课程项目足够

### 14.2 检索 embedding 备选

备选：

- `BAAI/bge-small-en-v1.5`

只有当第一版发现检索效果明显偏差，才补跑这一个备选。

### 14.3 检索实现方式

第一版固定：

- cosine similarity
- brute-force 检索

不一开始就上 `faiss`。

因为样本量目前并不大。

### 14.4 外部开源资源各自负责什么

- `LongMemEval`
  - 数据和任务来源
- `sentence-transformers`
  - chunk/query 向量表示
- `transformers`
  - 如果需要本地 answer module，可作为统一接口
- `faiss`
  - 仅在需要加速时引入
- `LightMem`
  - 借轻量 memory pipeline 拆分思路
- `HiMem`
  - 借层次记忆思路

### 14.5 我们自己的贡献边界

你们自己的工作重点应该写成：

- memory framework integration
- promotion score design
- routing rules
- baseline comparison pipeline

而不是说“我们发明了新的大模型”。

---

## 15. Answer generation 模型怎么定

### 15.1 先说结论

第一版不训练新的 answer model。

### 15.2 当前环境里可直接利用的接口线索

我已经看过你本机的 Codex 配置，当前能确认：

- 默认模型提供方：`custom`
- 当前默认模型：`vibe/gpt-5.4`
- 接口类型：OpenAI 兼容 `responses`
- base URL 已在本机配置中存在

这说明：

- 你现在**已经有一条现成可用的 API 路线**

### 15.3 第一版推荐的实验策略

分两阶段：

#### 阶段 A：主实验先看 retrieval

先固定主结果以 retrieval 为主：

- `Recall@6`
- `Hit@6`

理由：

- 先把你们真正的研究对象测清楚
- 避免生成模型能力掩盖记忆方法差异

#### 阶段 B：再接统一 answer model

answer generation 统一使用：

- `vibe/gpt-5.4`

理由：

- 当前环境现成可用
- 不需要你额外折腾新接口
- 所有 baseline 共用同一个回答模型，公平

### 15.4 统一回答 prompt

第一版固定 prompt：

```text
You are a QA model.
Answer the question only using the provided evidence.
If the evidence is insufficient, output exactly: Insufficient information.
Keep the answer within 20 words.
Do not explain your reasoning.

Question: {query}

Evidence:
{evidence_block}

Answer:
```

### 15.5 为什么不用“更聪明/联网”的模型做主实验变量

因为这个项目要测的是：

- 记忆管理差异
- 检索差异

不是“哪个大模型最强”。

所以 answer model 只能是：

- 固定
- 统一
- 不作为主要对比维度

---

## 16. 调参方法要具体到能立刻执行

这里直接定死，不再含糊。

### 16.1 默认起始参数

Round 0 初始值固定：

- `chunk size = 120`
- `overlap = 30`
- `STM capacity = 40`
- `LTM max size = 120`
- `consolidation frequency = 5`
- `promotion top-k = 3`
- `score threshold = 0.55`
- `final retrieval top-k = 6`

### 16.2 候选参数空间

只在下面这些候选值里调：

| 参数 | 候选值 |
| --- | --- |
| chunk size | `100`, `120`, `150` |
| overlap | `20`, `30`, `40` |
| STM capacity | `30`, `40`, `50` |
| LTM max size | `80`, `120`, `160` |
| consolidation frequency | `3`, `5`, `8` |
| promotion top-k | `2`, `3`, `5` |
| score threshold | `0.50`, `0.55`, `0.60` |
| final retrieval top-k | `4`, `6`, `8` |

### 16.3 调参算法

固定采用：

## **分阶段网格搜索（stage-wise grid search）**

意思是一次只调一小组参数，而不是所有组合暴力试完。

### 16.4 Round 1：先调切块

只调：

- chunk size
- overlap

其他参数全固定默认值。

评估优先级：

1. `Recall@6`
2. `Hit@6`
3. `Answer Accuracy`

保留原则：

- 先选 `Recall@6` 最好的
- 如果差距在 `1%` 以内，选 chunk 更少、实现更简单的那组

### 16.5 Round 2：调记忆容量

只调：

- STM capacity
- LTM max size

评估优先级：

1. `Recall@6`
2. `redundancy rate`
3. `retrieval latency`

保留原则：

- 在效果差不多时，优先 LTM 更小、重复率更低的组合

### 16.6 Round 3：调 promotion 参数

只调：

- consolidation frequency
- promotion top-k
- score threshold

评估优先级：

1. `Answer Accuracy`
2. `Recall@6`
3. `redundancy rate`

### 16.7 Round 4：调最终取几条证据

只调：

- final retrieval top-k

候选：

- `4`, `6`, `8`

保留原则：

- 如果 `top-8` 比 `top-6` 没明显提升，就固定 `6`
- 如果 `top-4` 损失明显，就不用它

### 16.8 怎么防止调参调傻

每一轮参数都必须同时满足：

- 在 `30` 条开发搜索集上表现不差
- 在 `20` 条开发确认集上也不塌

如果某组参数：

- 在 30 条很好
- 在 20 条掉很多

就视为不稳，不采用。

### 16.9 最终调参记录格式

必须保留一张表：

| Round | 调了什么 | 候选值 | 最终选什么 | 选择理由 |
| --- | --- | --- | --- | --- |

这张表以后可以直接放进报告。

---

## 17. 实验到底做多少次

这里也必须定死。

### 17.1 为什么要重复跑

因为：

- 抽样有随机性
- 开发集和正式实验集拆分也有随机性
- 如果只跑一次，很容易偶然

### 17.2 第一版实验重复次数

固定：

- 每个正式实验方法跑 `3` 次

只改变：

- 随机种子

种子固定为：

- `42`
- `43`
- `44`

### 17.3 正式报告怎么写结果

主表写：

- `mean ± std`

也就是：

- 平均值 ± 标准差

如果 3 次结果特别接近，说明方法比较稳。

### 17.4 开发调参阶段是否也跑 3 次

不需要全部 3 次。

开发调参阶段：

- 默认先跑 `1` 次
- 只有进入最后两组最优候选时，再补成 `3` 次确认

这样能省很多时间。

---

## 18. 评分和指标细节

### 18.1 主指标

最终主表只保留 3 个：

- `Answer Accuracy`
- `Recall@6`
- `Hit@6`

### 18.2 retrieval 指标怎么判

如果 `LongMemEval` 提供 gold supporting evidence 或 gold context：

- `Hit@6`：top-6 里是否至少命中一条 gold evidence
- `Recall@6`：top-6 覆盖了多少 gold evidence

### 18.3 answer 指标怎么判

优先级固定：

1. 如果数据集有官方评价方式：优先用官方方式
2. 如果有 gold answers：用标准化 exact match
3. 如果答案表达很自由：再增加一个固定 LLM judge 作为辅助分析

### 18.4 标准化 exact match 规则

如果自己写字符串判分，统一做：

- 全部小写
- 去基本标点
- 去多余空格
- 数字格式统一

然后比较。

### 18.5 LLM judge 的使用边界

如果后面需要用模型辅助判断，只允许：

- 作为补充分析

不允许：

- 作为唯一主评分指标

因为那样会让评分太依赖另一个模型。

### 18.6 分析指标

额外记录：

- `LTM size`
- `redundancy rate`
- `retrieval latency`

### 18.7 redundancy rate 怎么算

定义：

- LTM 中两两相似度 `>= 0.85` 的重复条目比例

### 18.8 retrieval latency 怎么算

定义：

- 单次 query 从开始检索到返回 top-k evidence 的平均耗时

---

## 19. 消融实验到底做什么

消融就是：

> 把你方法里的一个零件拆掉，看结果会不会变差

### 19.1 固定做这 3 个

1. 去掉 `relevance`
2. 去掉 `diversity`
3. 去掉 query-aware routing

### 19.2 具体怎么做

#### Ablation A

把公式改成：

`score = 0.6 * reuse + 0.4 * diversity`

#### Ablation B

把公式改成：

`score = 0.5 * relevance + 0.5 * reuse`

#### Ablation C

不用 Rule 1/2/3，统一强制使用：

- `Naive STM+LTM` 检索路径

### 19.3 做消融的目的

要证明：

- 不是系统“大概有用”
- 而是你们方法里的具体部件真的有贡献

---

## 20. 从 0 开始的实际执行顺序

下面这部分就是接下来真正要按顺序做的事情。

### Step 1：确认 LongMemEval 数据结构

要做：

- 确认下载方式
- 确认字段
- 确认是否有 gold answer
- 确认是否有 gold evidence

产出：

- 一份简短数据结构说明
- 一段样本打印脚本

验收标准：

- 能打印 `10` 条样本
- 人能看懂 `context / query / answer`

### Step 2：搭代码骨架

要做：

- 建 `data/`
- 建 `experiments/`
- 建 `results/`
- 建 `src/`

产出：

- 最小可运行项目结构

验收标准：

- 主入口能跑通空流程

### Step 3：实现数据读取和切块

要做：

- `data_loader.py`
- `chunking.py`

产出：

- 每条样本对应的 chunk 列表

验收标准：

- 任意样本能被稳定切块

### Step 4：实现 Flat Memory

要做：

- embedding
- 相似度
- top-k evidence

产出：

- 第一版 retrieval 结果

验收标准：

- 至少 `20` 条样本可跑通

### Step 5：实现评测

要做：

- `Hit@6`
- `Recall@6`
- 如可行则 `Answer Accuracy`

产出：

- 第一版结果数字

### Step 6：实现 STM/LTM 结构

要做：

- `MemoryItem`
- STM
- LTM
- 容量控制

产出：

- 记忆状态可打印

### Step 7：实现结构 baseline

要做：

- STM-only
- LTM-only
- Naive STM+LTM

产出：

- 第一张结构 baseline 表

### Step 8：实现 Frequency-based Promotion

要做：

- reuse 统计
- top-3 写入

产出：

- promotion baseline 结果

### Step 9：实现 Ours

要做：

- relevance
- reuse
- diversity
- Rule 1/2/3

产出：

- SMC-QA 主结果

### Step 10：严格按调参流程调参

要做：

- Round 1 到 Round 4

产出：

- 参数冻结版配置

### Step 11：正式实验

要做：

- 在 150 条正式实验集上跑 `3` 次

产出：

- 主结果表

### Step 12：消融和案例

要做：

- 3 个消融
- 3 个成功案例
- 3 个失败案例

产出：

- 消融表
- 案例分析

---

## 21. 建议的代码结构

```text
ml_prejects/
├── data/
├── experiments/
├── src/
│   ├── data_loader.py
│   ├── chunking.py
│   ├── memory.py
│   ├── retriever.py
│   ├── promotion.py
│   ├── router.py
│   ├── pipeline.py
│   └── evaluate.py
├── results/
├── guidelines/
└── memory/
```

---

## 22. 风险和 fallback

### 22.1 风险 1：LongMemEval 接入太慢

处理：

- 先缩到单子任务
- 再缩样本数

### 22.2 风险 2：answer generation 波动太大

处理：

- retrieval 指标作为主结果
- answer generation 只做补充

### 22.3 风险 3：summary-based baseline 做不完

处理：

- 第一版直接不做

### 22.4 风险 4：调参太慢

处理：

- 先 1 次跑开发候选
- 只有最终候选再跑 3 次

### 22.5 风险 5：ours 提升不明显

处理：

- 重点写分析
- 重点写 memory 行为差异
- 课程项目不是只有“赢很多”才算成功

---

## 23. 当前最终结论

这份方案就是当前最推荐的执行版本：

- 和已提交 proposal 主线一致
- 满足课程 Applied / Engineering 要求
- 用真实数据
- 至少 5 个 baseline/对照方法
- 有清楚的调参方法
- 有清楚的模型选择
- 有清楚的评分方式
- 有清楚的实验轮次
- 有清楚的 fallback

也就是说，到这里为止，项目已经不再只是“想法”，而是一个可以直接开始搭代码和跑实验的方案。
