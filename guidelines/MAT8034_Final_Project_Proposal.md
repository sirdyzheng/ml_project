# Final Project Proposal

## Title

**SMC-QA: Selective Memory Consolidation with Query-Aware Retrieval for Long-Context QA**

## Category

**2 Applied / Engineering**

## Team Members

- **Zheng Yifan** — 12532245
- **Chen Zeming** — 12531410

## Abstract

Long-context question answering requires models to use past information effectively while avoiding redundancy and noise. Recent studies show that hierarchical memory and memory consolidation are promising directions for long-horizon interaction, but many existing systems are relatively complex and difficult to reproduce in a course project setting. In this project, we propose **SMC-QA**, a lightweight hierarchical memory framework with short-term memory (STM) and long-term memory (LTM). We will compare several representative baselines, including flat memory, STM-only retrieval, LTM-only retrieval, naive STM+LTM merging, frequency-based promotion, and summary-based promotion. We further explore a simple selective promotion strategy based on relevance, reuse frequency, and information diversity, together with query-aware retrieval across memory levels. We will evaluate these methods on a public long-context QA benchmark and analyze both retrieval quality and final QA performance.

## Motivation

Long-context QA is challenging because useful evidence is often mixed with irrelevant, repeated, or outdated information. A flat memory structure may therefore introduce substantial noise during retrieval. Recent work has shown that hierarchical memory, multi-granularity summaries, and long-term memory consolidation can improve long-horizon reasoning, but these systems are often fairly heavy and involve complicated pipelines. Our goal is to study this problem in a simpler and more reproducible setting: instead of building a large memory system from scratch, we focus on one key question: **how should short-term memory be promoted into long-term memory, and how should retrieval use these two memory levels?** [1]

## Method

We build a lightweight hierarchical external memory framework for long-context QA. Incoming information is first stored in **short-term memory (STM)**. A subset of STM entries may later be promoted into **long-term memory (LTM)** through a consolidation rule. At inference time, retrieval can access one or both memory levels depending on the query.

Our main idea is **selective memory consolidation**. Instead of promoting all STM content into LTM, we assign each memory item a simple score based on three signals:

- **Relevance**: how related the memory item is to the current or recent query context
- **Reuse**: how often the memory item is retrieved or reused later
- **Diversity**: whether the item contributes new information rather than duplicating existing LTM content

We also use **query-aware retrieval**, where the system can prioritize STM, LTM, or both, depending on the type of query. This keeps the method lightweight while still going beyond a fixed flat-memory baseline [2].

## Related Work

Our project is mainly inspired by recent work on long-term and hierarchical memory for LLM-based systems. **LongMemEval** provides a benchmark for evaluating long-term interactive memory, including information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention [1]. **HiMem** studies hierarchical long-term memory for long-horizon dialogue and supports both hierarchical memory construction and hybrid retrieval [2]. **Reflective Memory Management (RMM)** summarizes interactions across multiple granularities such as utterances, turns, and sessions, which motivates our summary-based promotion baseline [3]. **LightMem** modularizes memory into retrieval, writing, and long-term consolidation, showing that lightweight decomposition is a practical design direction [4].

Compared with these works, our project does not aim to build a full production-scale memory system. Instead, we focus on a smaller and more reproducible comparative study suitable for a course project, centered on promotion strategies and simple query-aware retrieval.

## Baselines

We plan to compare the following baselines:

1. **Flat Memory**All memory items are stored in one unified memory pool.
2. **STM-only Retrieval**The system retrieves only from short-term memory.
3. **LTM-only Retrieval**The system retrieves only from long-term memory.
4. **STM+LTM Naive Merge**The system retrieves from both memory levels and directly merges the results.
5. **Frequency-based Promotion**STM items are promoted to LTM based on how often they are reused or retrieved.
6. **Summary-based Promotion**STM content is summarized before being stored in LTM, inspired by multi-granularity memory management [3].
7. **SMC-QA (ours)**
   A selective promotion strategy based on relevance, reuse, and diversity, combined with query-aware retrieval.

## Intended Experiments

We plan to evaluate all methods on a public long-context memory benchmark, preferably **LongMemEval** [1]. Our experiments will include three parts.

First, we will conduct a **main comparison** among all baselines and SMC-QA, focusing on final QA accuracy and retrieval effectiveness.

Second, we will perform a small **ablation study** on our promotion rule by removing one factor at a time, such as relevance, reuse, or diversity, to understand which factor contributes most.

Third, we will provide a brief **analysis** of how the memory system behaves, such as:

- whether STM or LTM is used more often for different query types,
- whether selective promotion reduces noisy long-term memory,
- and whether query-aware retrieval improves evidence selection.

## Expected Outcome

We expect this project to provide:

- a simple and reproducible hierarchical memory framework for long-context QA,
- a systematic comparison of representative promotion baselines,
- and a lightweight selective promotion strategy that may improve both retrieval quality and downstream QA performance.

## References

1. Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory**. *arXiv preprint arXiv:2410.10813*, 2024. https://arxiv.org/abs/2410.10813
2. Zhang, N., Yang, X., Tan, Z., Deng, W., and Wang, W. **HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents**. *arXiv preprint arXiv:2601.06377*, 2026. https://arxiv.org/abs/2601.06377
3. Tan, Z., Yan, J., Hsu, I.-H., Han, R., Wang, Z., Le, L., Song, Y., Chen, Y., Palangi, H., Lee, G., Iyer, A. R., Chen, T., Liu, H., Lee, C.-Y., and Pfister, T. **In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents**. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 8416-8439, 2025. https://aclanthology.org/2025.acl-long.413/
4. Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y., Xu, Z., Deng, S., Yao, Y., Wang, M., Qiao, S., Chen, H., and Zhang, N. **LightMem: Lightweight and Efficient Memory-Augmented Generation**. *arXiv preprint arXiv:2510.18866*, 2025. https://arxiv.org/abs/2510.18866