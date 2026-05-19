# Measuring a Language Model（Perplexity 详解）


## 一、文章核心主题

这是一篇**深度讲解 Perplexity（困惑度）的技术文章**，从信息论基础出发，完整推导 perplexity 与 cross-entropy、bits-per-byte 之间的关系，并在 PyTorch 中用 GPT-2 实际计算演示。


## 二、核心概念

### 2.1 语言模型的基本任务

**语言模型**的核心任务是：**给定前文，预测下一个 token 的概率分布**。

- 输入：`In a shocking finding, scientist discovered a herd of unicorns`
- 模型输出：对所有 50257 个可能的下一个 token 输出概率分布
- 例如：`perfect` 之后的下一个词，GPT-2 给出：

```
64.57%   English
17.82%   Spanish
2.26%    Latin
2.02%    ,
1.81%    human
```

### 2.2 什么是 Perplexity？

**Perplexity = 2^{cross-entropy}**

衡量模型在给定文本上的"困惑程度"——模型对文本的预测有多不确定。

$$
\text{PPL} = 2^{\frac{1}{N} \sum_{i=1}^{N} -\log_2 P(t_i | t_1, ..., t_{i-1})}
$$

- **PPL = 2**：模型对下一个 token 的预测"二选一"，完全不确定
- **PPL = 1**：模型完美预测，完全确定
- **PPL 越低越好**

### 2.3 Perplexity 与 Cross-Entropy 的关系

**Cross-entropy** 是平均每个 token 需要的编码位数：

$$
H = -\frac{1}{N} \sum_i \log P(t_i | C)
$$

Perplexity 是 cross-entropy 的指数形式：

$$
\text{PPL} = 2^{H}
$$

### 2.4 Perplexity 与压缩的关系

Cross-entropy 给出了**理论最优压缩的下界**（香农信息论）：

- 用 bit（以 2 为底）：$H_{\text{bits}} = H_{\text{nats}} / \ln 2$
- 实际可用**算术编码（Arithmetic Coding）**逼近这个下界

> 核心洞察：语言模型的预测能力 = 压缩能力。更好的语言模型能更好地"压缩"文本，perplexity 更低。

---

## 三、PyTorch 实现要点

### 3.1 核心代码流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# 加载 GPT-2 XL (1.5B 参数)
model_name = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
text = "In a shocking finding, scientist discovered a herd of unicorns..."
tokens = tokenizer(text, return_tensors='pt')['input_ids']

# 前向传播：得到每个位置的 logits
with torch.inference_mode():
    logits = model(tokens).logits.cpu()

# softmax 得到概率
probs = F.softmax(logits, dim=-1)

# 计算每个 token 的对数概率
# 目标 token 是原始序列左移一位
target_tokens = tokens  # 偏移一位
logprobs = torch.gather(probs.log(), dim=-1, index=target_tokens.unsqueeze(-1)).squeeze()

# 平均对数似然
avg_logprob = logprobs.mean()
ppl = torch.exp(avg_logprob)
```

### 3.2 数值稳定性技巧

概率的连乘容易**下溢**（underflow），解决方法是先取对数：

```python
# ❌ 危险：连乘会下溢
prob = (p1 * p2 * p3 * ...).item()  # → 0

# ✅ 安全：取对数后求和
log_prob = log(p1) + log(p2) + log(p3) + ...
prob = exp(log_prob)
```

同时在 softmax 中减去最大值防止指数溢出：

```python
logits - logits.max(dim=-1, keepdim=True).values  # 数值稳定版
```

---

## 四、重要结论

### 4.1 Perplexity 能对比吗？

- ✅ 可以跨模型架构对比（GPT-2、Llama、Mistral 等）
- ❌ 不能直接用于 **BERT / T5** 等 Masked LM（需要用 pseudo-log likelihood）
- ⚠️ HuggingFace 的 WikiText-2 perplexity 计算**有 bug**（按 token 切分而非按词）

### 4.2 Perplexity 与下游任务的关系

| 关系 | 说明 |
|------|------|
| ✅ 某些设置下对齐 | perplexity 低 → 下游任务好 |
| ❌ 并非所有设置 | perplexity 低不一定任务好 |
| ✅ 可用于数据选择 | 用 perplexity 筛选高质量训练数据 |

### 4.3 一个令人震惊的数字

用 GPT-2 生成 unicorn 故事的概率约为 $4.2 \times 10^{-60}$。

如果有一百万个语言模型，每个每秒生成一百万个 token，产生这段文字的**预期时间**：

$$
\approx 7.5 \times 10^{41} \text{ 年}
$$

宇宙年龄才约 $10^{10}$ 年，所以这段文字**几乎不可能随机生成**——这恰恰说明了 GPT-2 确实学到了语言的结构。

---

## 五、关键公式汇总

| 概念                | 公式                                  | 单位        |            |
| ----------------- | ----------------------------------- | --------- | ---------- |
| **序列概率**          | $P(t_1,...,t_n) = \prod_i P(t_i     | C_i)$     | -          |
| **对数似然**          | $\log P = \sum_i \log P(t_i         | C_i)$     | nats       |
| **Cross-Entropy** | $H = -\frac{1}{N} \sum_i \log P(t_i | C_i)$     | nats/token |
| **Perplexity**    | $\text{PPL} = 2^H$                  | -         |            |
| **Bits-per-byte** | $H / \ln 2 / \text{bytes}$          | bits/byte |            |

---

## 六、实用资源

| 资源 | 说明 |
|------|------|
| [HuggingFace perplexity docs](https://huggingface.co/docs/transformers/perplexity) | ⚠️ 注意 WikiText-2 计算有 bug |
| [lm_perplexity](https://github.com/) | 推荐替代工具 |
| [Chip Huyen - Understanding Evaluation Metrics](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/) | Perplexity 入门概述 |
| [LAMBADA 数据集](https://arxiv.org/abs/1606.06031) | 长距离依赖评估基准 |
| [Enwik8](https://mattmahoney.net/text/enwik8.html) | 压缩评估标准数据集 |

---

## 七、一句话总结

> **Perplexity 是衡量语言模型预测能力的核心指标，本质是 cross-entropy 的指数形式，连接了语言建模与信息论压缩。模型预测越准，perplexity 越低，压缩率越高。**
