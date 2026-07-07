---
layout: listpage
title: LLM
subtitle: 大语言模型、RAG、推理优化
article-list:
  - article-title: 注意力机制（Attention）
    article-url: /llm/attention
    article-date: 2026-07-07
    article-desc: 详解 Scaled Dot-Product Attention, Multi-Head Attention (MHA), Multi-Query Attention (MQA)及Grouped-Query Attention (GQA)四种经典机制。
    article-tags: [Atttention, MHA, MQA, GQA]
  - article-title: 位置编码（Positional Encoding）
    article-url: /llm/positional_encoding
    article-date: 2026-07-06
    article-desc: 详解 Learnable（Bahdanau et al., 2014）、Sinusoidal（Vaswani et al., 2017）、RoPE（Su et al., 2022）及YaRN（Peng et al., 2023）四种主流位置编码方案的数学原理与代码实现，梳理从绝对位置编码到相对位置编码的演进脉络，深度解析 RoPE如何通过旋转矩阵实现相对位置感知，以及YaRN如何通过频率分段与温度修正实现长上下文窗口扩展。 。
    article-tags: [Sinusoidal, RoPE, YaRN]
  - article-title: 字节对编码(BPE):一种简单而高效的开放词表分词方法
    article-url: /dl/bpe
    article-date: 2026-06-17
    article-desc: 从最小的符号单元出发,反复将训练语料中共现频率最高的相邻符号对合并为一个新符号,直至词表达到预设规模。
    article-tags: [BPE, tokenizer]
---
