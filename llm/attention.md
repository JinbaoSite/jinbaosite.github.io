# Transformer 注意力机制研究综述

注意力机制（Attention）作为 Transformer 架构的核心组件，自提出以来一直是自然语言处理领域的研究热点。本文围绕 Transformer 中的四种注意力机制展开综述：Scaled Dot-Product Attention、Multi-Head Attention、Multi-Query Attention 以及 Grouped-Query Attention。

## 1. Scaled Dot-Product Attention

Scaled Dot-Product Attention 最早由 Vaswani 等人在论文《Attention Is All You Need》（2017）中提出，是 Transformer 架构的基础计算单元。该机制的核心思想源于早期的神经机器翻译工作，通过计算查询向量与键向量的点积相似度来确定注意力权重，进而对值向量进行加权求和。

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

| 矩阵                      | 含义            | 矩阵维度 (单样本批量)   |
|:------------------------|:--------------|:---------------|
| $Q$                     | 查询矩阵 (Query)  | $n \times d_k$ |
| $K$                     | 键矩阵 (Key)     | $n \times d_k$ |
| $V$                     | 值矩阵 (Value)   | $n \times d_v$ |
| $QK^T$                  | 原始注意力分数       | $n \times n$   |
| $\text{Softmax}(\cdot)$ | 注意力权重分布       | $n \times n$   |
| **Output**              | **注意力机制最终输出** | $n \times d_v$ |

### 1.1 核心要素定义

假设输入序列包含 $n$ 个向量，每个向量的特征维度为 $d_{\text{model}}$。
在计算 Attention 之前，输入矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ 会通过三个不同的线性变换矩阵（权重矩阵），映射得到三个核心矩阵：

* **Query (查询矩阵) $Q$**：$Q = X W_Q$ （其中 $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$）
* **Key (键矩阵) $K$**：$K = X W_K$ （其中 $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$）
* **Value (值矩阵) $V$**：$V = X W_V$ （其中 $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$）

> **注**：在标准的 Transformer 中，通常令 $d_k = d_v = d_{\text{model}}  / \text{num}_{\text{heads}}$。

### 1.2 详细计算步骤

Attention 的计算主要分为以下 5 个步骤：

#### 1.2.1 Step 1: 计算原始注意力分数（Scores）

通过将 $Q$ 和 $K$ 的转置进行矩阵乘法，计算出 Query 和 Key 之间的相似度。点积结果越大，说明两个向量的相关性越高。

$$\text{Scores} = Q K^T$$

* **维度变化**：$(n \times d_k) \times (d_k \times n) \rightarrow (n \times n)$
* 结果矩阵中的第 $(i, j)$ 个元素，代表第 $i$ 个单词对第 $j$ 个单词的原始注意力得分。

#### 1.2.2 Step 2: 缩放操作（Scaling）

将原始分数除以 $\sqrt{d_k}$（即 Key 向量维度的平方根）。

$$\text{Scaled Scores} = \frac{Q K^T}{\sqrt{d_k}}$$

* **目的**：当 $d_k$ 很大时，点积的结果会非常大，导致后面经过 Softmax 函数时梯度变得极小（进入饱和区）。除以 $\sqrt{d_k}$ 可以起到方差缩放的作用，让训练更加稳定。

#### 1.2.3 Step 3: 掩码操作（Mask）

对缩放后的分数应用掩码（Mask），将需要屏蔽的位置设置为负无穷（$-\infty$），使其在后续 Softmax 计算后的权重趋近于 0。

$$\text{Masked Scores} = \text{Mask}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

* **目的**：
  * **Padding Mask**：屏蔽输入序列中的 Padding 部分（通常是因为 batch 内句子长度不同，填充的无效 tokens）
  * **Causal Mask / Sequence Mask**：在 Decoder 中，防止当前位置看到后续位置的信息（确保自回归特性）
* **实现方式**：将需要屏蔽的位置乘以一个非常大的负数（如 $-1e9$ 或 $-\infty$），Softmax 会对这些位置输出接近 0 的权重

#### 1.2.4 Step 4: 归一化（Softmax）

对掩码处理后的分数在**行方向**上应用 Softmax 函数，将其转化为概率分布（所有权重相加为 1）。

$$\text{Attention Weights} = \text{Softmax}\left(\text{Mask}\left(\frac{Q K^T}{\sqrt{d_k}}\right)\right)$$

* **维度**：$(n \times n)$
* 此时矩阵里的数值就是最终的**注意力权重**（矩阵中每行的和为 1）。

#### 1.2.5 Step 5: 加权求和（Output）

用计算出的注意力权重矩阵去乘以 Value 矩阵 $V$，实现对 Value 的加权聚合。

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\text{Mask}\left(\frac{Q K^T}{\sqrt{d_k}}\right)\right) V$$

* **维度变化**：$(n \times n) \times (n \times d_v) \rightarrow (n \times d_v)$
* 最终输出的矩阵中，每一个向量都融合了整个序列中与其相关的其他向量的信息。

### 1.3 代码实现

```python
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any

class ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    完整计算公式：
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

    """
    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, 
                Q: Tensor, 
                K: Tensor, 
                V: Tensor, 
                mask: Tensor = None, 
                return_attention_socre: bool = False
    ) -> tuple[Tensor, Any] | Tensor:
        """
        :param Q: Tensor, shape = (batch_size, num_heads, seq_len, d_k) - 查询矩阵
        :param K: Tensor, shape = (batch_size, num_heads, seq_len, d_k) - 键矩阵
        :param V: Tensor, shape = (batch_size, num_heads, seq_len, d_v) - 值矩阵 (通常 d_k == d_v)
        :param mask: Tensor, shape = (batch_size, num_heads, seq_len, seq_len] - 掩码矩阵（可选，用于Transformer中的Padding或Decoder的Causal Mask）
        :param return_attention_socre: 是否返回注意力分数
        :return:
            output: Tensor, shape = (batch_size, num_heads, seq_len, d_k)
        """
        d_k = Q.shape[-1] # 向量的维度,即d_k
        # STEP 1: 计算原始注意力分数 \frac{QK^T}{\sqrt(d_k)}
        scores = torch.matmul(Q, K.transpose(-1, -2))
        # STEP 2: 缩放
        scores = scores / torch.sqrt(torch.tensor(d_k, device=Q.device))
        # STEP 3: Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # STEP 4: 归一化softmax 得到注意力权重
        attention_scores = F.softmax(scores, dim=-1)
        # STEP 5: 加权求和
        output = torch.matmul(attention_scores, V)
        if return_attention_socre:
            return output, attention_scores
        return output
```

在 Vaswani 等人提出 Scaled Dot-Product Attention 之后，同一论文中进一步引入了多头注意力机制（Multi-Head Attention），以增强模型的表达能力。

## 2. Multi-Head Attention

**Multi-Head Attention（多头注意力机制）** 是在 Scaled Dot-Product Attention（缩放点积注意力）的基础上发展而来的。

它的核心思想是：**与其只计算一次复杂的注意力（单头），不如将输入拆分为多个低维的子空间（多个头），在每个子空间内独立地计算注意力，最后再将所有头的输出拼接并线性映射回去。** 这样做能让模型同时关注来自不同位置、不同表示子空间的信息。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{where } \text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

### 2.1 核心要素定义

假设输入序列矩阵为 $X \in \mathbb{R}^{n \times d_{\text{model}}}$（其中 $n$ 为序列长度，$d_{\text{model}}$ 为模型的隐藏层维度），我们要计算 $h$ 个头的注意力。

在多头注意力中，我们需要 $h$ 组不同的线性变换矩阵。对于第 $i$ 个头（$i = 1, 2, \dots, h$），其权重矩阵分别定义为：

* $W_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* $W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$

通常，为了保持总参数量和计算量与单头一致，我们会令 $d_k = d_v = d_{\text{model}} / h$。

### 2.2 详细计算步骤

Multi-Head Attention 的计算主要分为以下 4 个步骤：

#### 2.2.1 Step 1: 线性映射（Linear Projection）

将输入矩阵 $X$ 分别乘以 $h$ 组不同的权重矩阵，得到每个头专属的 $Q_i, K_i, V_i$ 矩阵：

$$Q_i = X W_Q^{(i)}, \quad K_i = X W_K^{(i)}, \quad V_i = X W_V^{(i)}$$

* **维度**：每个头的 $Q_i, K_i$ 维度为 $(n \times d_k)$，$V_i$ 维度为 $(n \times d_v)$。

#### 2.2.2 Step 2: 独立计算每个头的注意力（Scaled Dot-Product Attention）

每个头独立运行标准的缩放点积注意力机制，得到该头的输出 $\text{head}_i$：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

* **维度**：$\text{head}_i \in \mathbb{R}^{n \times d_v}$

#### 2.2.3 Step 3: 拼接（Concat）

将所有 $h$ 个头计算出来的输出矩阵在特征维度（列方向）上横向拼接在一起：

$$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)$$

* **维度变化**：由于有 $h$ 个维度为 $(n \times d_v)$ 的矩阵拼接，拼接后的总维度变为 $(n \times (h \times d_v))$。
* 因为 $h \times d_v = d_{\text{model}}$，所以拼接后的矩阵维度重新恢复到了 $(n \times d_{\text{model}})$。

#### 2.2.4 Step 4: 最后的线性变换（Output Linear）

为了让多头聚集起来的信息能够充分融合，拼接后的结果会通过一个最终的输出权重矩阵 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 进行线性映射：


$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

* **维度**：$(n \times d_{\text{model}}) \times (d_{\text{model}} \times d_{\text{model}}) \rightarrow (n \times d_{\text{model}})$

### 2.3 矩阵层面的高效实现（并行化）

在实际工程实现中，**并不会**真的用 `for` 循环去一个头一个头地计算。标准的并行化做法是：

1. **一次性投影**：直接用一个大矩阵将 $X$ 映射成总维度为 $(n \times d_{\text{model}})$ 的 $Q, K, V$。
2. **维度重塑（Reshape）**：将维度从 $(n \times d_{\text{model}})$ 改为 $(n \times h \times d_k)$。
3. **维度置换（Transpose）**：转换为 $(\text{batch_size}, h, n, d_k)$。
4. **批量矩阵乘法（BMM）**：利用高维矩阵乘法，一条指令同时计算所有样本、所有头的注意力，极大提升了 GPU 的运行效率。

### 2.4 代码实现

```python 
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, n_heads: int, d_model: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads # 注意力头的数量
        self.d_model = d_model # 模型的总维度
        # 每个 head 的维度
        self.d_k = d_model // n_heads
        # 线性映射：Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 输出映射
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaleDotProductAttention()

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None
    ):
        """
        :param Q: Tensor, shape = (batch_size, seq_len, d_model)
        :param K: Tensor, shape = (batch_size, seq_len, d_model)
        :param V: Tensor, shape = (batch_size, seq_len, d_model)
        :param mask: Tensor, shape = (batch_size, seq_len, d_model)
        :return:
            output: Tensor, shape = (batch_size, seq_len, d_model)
        """
        batch_size = Q.shape[0]
        # STEP 1: 线性变换 将原始的 Q, K, V 映射到新的特征空间
        q_proj = self.W_q(Q) # (batch_size, seq_len, d_model)
        k_proj = self.W_k(K) # (batch_size, seq_len, d_model)
        v_proj = self.W_v(V) # (batch_size, seq_len, d_model)
        # STEP 2: reshape + transpose 拆成多头 
        # 形状从 (batch_size, seq_len, d_model) 变为 (batch_size, n_heads, seq_len, d_k)
        q_heads = q_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = k_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = v_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # STEP 3: 调用attention (batch_size, n_heads, seq_len, d_k)
        context = self.attention(q_heads, k_heads, v_heads, mask=mask)  
        # STEP 4: 拼接多头
        context = context.transpose(1, 2).contiguous()
        output = context.view(batch_size, -1, self.d_model)
        # STEP 5: 线性映射
        output = self.W_O(output)
        return output
```

尽管 Multi-Head Attention 在建模能力上表现优异，但随着模型规模增大和序列长度增加，其推理效率问题日益突出。针对这一问题，Shazeer 等人在论文《Fast Transformer Decoding: One Write-Head is All You Need》（2019）中提出了 Multi-Query Attention（MQA），通过让多个 Query 头共享同一组 Key 和 Value 来显著降低推理时的 KV Cache 开销。

## 3. Multi-Query Attention

**Multi-Query Attention (MQA)** 是由 Noam Shazeer 在 2019 年提出的一种 Attention 变体。

它的核心动机是为了**解决大模型推理（Generation）时的性能瓶颈**。在传统的 Multi-Head Attention (MHA) 中，每个独立头都有自己的 $Q, K, V$ 矩阵。在推理时，为了避免重复计算，模型会把历史 Token 的 $K$ 和 $V$ 缓存起来（即 **KV Cache**）。随着序列变长，KV Cache 会吞噬极大的显存带宽和空间。

MQA 的核心思想极其精简：**让所有的 Query 头共享同一组 Key 和 Value 头。**

### 3.1 核心要素定义

假设输入序列矩阵为 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，我们要计算 $h$ 个 Query 头。

与 MHA 为每个头都准备独立的 $W_Q, W_K, W_V$ 不同，MQA 只准备：

* $h$ 组不同的 Query 权重矩阵：$W_Q^{(1)}, W_Q^{(2)}, \dots, W_Q^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* **仅仅 1 组** Key 权重矩阵：$W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* **仅仅 1 组** Value 权重矩阵：$W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

通常，每个头的维度仍满足 $d_k = d_v = d_{\text{model}} / h$。

### 3.2 详细计算步骤

MQA 的计算步骤与标准注意力机制相似，但由于 $K$ 和 $V$ 的共享特性，矩阵的维度和广播（Broadcasting）逻辑发生了变化：

#### 3.2.1 Step 1: 线性映射（Linear Projection）

投影时，Query 依然有 $h$ 个不同的结果，而 $K$ 和 $V$ 只有单份：

* 对于第 $i$ 个 Query 头：$Q_i = X W_Q^{(i)} \quad \in \mathbb{R}^{n \times d_k}$
* 公共的 Key 矩阵：$K = X W_K \quad \in \mathbb{R}^{n \times d_k}$
* 公共的 Value 矩阵：$V = X W_V \quad \in \mathbb{R}^{n \times d_v}$

#### 3.2.2 Step 2: 独立计算每个 Query 头的注意力

每个独立的 Query 头 $Q_i$ 都去和**同一个** $K$ 计算注意力分数，并和**同一个** $V$ 进行加权求和。

$$\text{head}_i = \text{Attention}(Q_i, K, V) = \text{Softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

* **维度**：$\text{head}_i \in \mathbb{R}^{n \times d_v}$
* **注意**：这里的 $K^T$ 和 $V$ 对所有的 $\text{head}_i$ 来说是完全一模一样的。

#### 3.2.3 Step 3: 拼接（Concat）

将这 $h$ 个头得到的输出矩阵在特征维度上横向拼接：

$$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)$$

* **维度**：$h \times (n \times d_v) \rightarrow (n \times d_{\text{model}})$

#### 3.2.4 Step 4: 最后的线性变换（Output Linear）

与 MHA 一样，通过一个输出权重矩阵 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 融合多头信息：

$$\text{MultiQuery}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

* **维度**：$(n \times d_{\text{model}})$

### 3.3 完整公式总结

$$\text{MultiQuery}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{where } \text{head}_i = \text{Softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

虽然数学公式看起来和 MHA 极其相似，但请注意其本质区别：**在 $\text{head}_i$ 的计算中，所有的 $i$ 共享相同的 $K$ 和 $V$。**

### 3.4 并行化矩阵维度的变化对照

在现代深度学习框架中，通常会利用张量的广播机制（Broadcasting）来实现 MQA 的并行化计算。可以通过下表清晰地看出 MHA 与 MQA 在张量维度上的断层式差异：

| 矩阵 / 张量 | MHA (多头注意力) 维度 | MQA (多查询注意力) 维度 |
| --- | --- | --- |
| **Query ($Q$)** | `[batch_size, num_heads, seq_len, d_k]` | `[batch_size, num_heads, seq_len, d_k]` |
| **Key ($K$)** | `[batch_size, num_heads, seq_len, d_k]` | `[batch_size, 1, seq_len, d_k]` |
| **Value ($V$)** | `[batch_size, num_heads, seq_len, d_v]` | `[batch_size, 1, seq_len, d_v]` |
| **Score Matrix** | `[batch_size, num_heads, seq_len, seq_len]` | `[batch_size, num_heads, seq_len, seq_len]` *(通过广播计算)* |

### 3.5 为什么 MQA 能极大加速推理？

1. **减少显存占用**：在推理时，KV Cache 的大小缩减到了原来的 $1 / h$（如果模型有 32 个头，KV 显存直接缩减 32 倍）。
2. **打破带宽瓶颈**：大模型自回归生成时，瓶颈不在于 GPU 计算力（Math-bound），而在于 GPU 内存带宽（Memory-bound）。由于每次读取的 $K, V$ 变少，GPU 可以花更少的时间去搬运内存，从而大幅提升 Token 的生成速度。

### 3.6 代码实现

```python 
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any

class MultiQueryAttention(nn.Module):
    """
    Multi Query Attention 
    """
    def __init__(self, n_heads: int, d_model: int):
        super(MultiQueryAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.n_heads
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_k, bias=False)
        self.attention = ScaleDotProductAttention()
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        :param Q: Tensor, shape = (batch_size, seq_len, d_model)
        :param K: Tensor, shape = (batch_size, seq_len, d_model)
        :param V: Tensor, shape = (batch_size, seq_len, d_model)
        :param mask: Tensor, shape = (batch_size, 1, seq_len, seq_len)
        :return:
        """
        batch_size = Q.shape[0]
        # 1. 线性映射 (Linear Projection)
        q_proj = self.W_q(Q)
        k_proj = self.W_k(K)
        v_proj = self.W_v(V)
        # 2. 维度重塑与变换 (Reshape & Transpose)
        # Q: [batch_size, len_q, d_model] -> [batch_size, num_heads, len_q, d_k]
        q_heads = q_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K 和 V 增加一个 Head 维度（设为 1 即可），以便后续进行广播计算
        # [batch_size, len_k, d_k] -> [batch_size, 1, len_k, d_k]
        k_heads = k_proj.unsqueeze(1)
        v_heads = v_proj.unsqueeze(1)
        # STEP 3: 调用attention (batch_size, n_heads, seq_len, d_k)
        context = self.attention(q_heads, k_heads, v_heads, mask=mask)  
        # STEP 4: 拼接多头
        context = context.transpose(1, 2).contiguous()
        output = context.view(batch_size, -1, self.d_model)
        # STEP 5: 线性映射
        output = self.W_O(output)
        return output
```

MQA 虽然有效降低了推理开销，但过度共享 KV 导致模型表达能力下降。为解决这一问题，Ainslie 等人在论文《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》（2023）中提出了 Grouped-Query Attention（GQA），通过引入分组机制在 MHA 和 MQA 之间取得平衡。

## 4. Grouped-Query Attention

**Grouped-Query Attention (GQA)** 是由 Ainslie 等人在 2023 年提出的一种 Attention 变体。

它是 **Multi-Head Attention (MHA)** 和 **Multi-Query Attention (MQA)** 的折中方案。MHA 虽然表达能力强，但推理时 KV Cache 显存开销巨大；MQA 虽然极大地压缩了 KV Cache，但由于所有头强行共享一组 KV，会导致模型表达能力下降。

GQA 的核心思想是：**将 Query 头分成若干个组（Groups），每一个组内的所有 Query 头共享同一组 Key 和 Value 头。**


### 4.1 核心要素定义

假设输入序列矩阵为 $X \in \mathbb{R}^{n \times d_{\text{model}}}$。我们设定：

* Query 的总头数为 $h$。
* Key 和 Value 的总头数为 $g$（即分组数）。
* 每个组内包含的 Query 头数为 $m = h / g$。

在 GQA 中，权重矩阵的定义如下：

* $h$ 组不同的 Query 权重矩阵：$W_Q^{(1)}, W_Q^{(2)}, \dots, W_Q^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* $g$ 组不同的 Key 权重矩阵：$W_K^{(1)}, W_K^{(2)}, \dots, W_K^{(g)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
* $g$ 组不同的 Value 权重矩阵：$W_V^{(1)}, W_V^{(2)}, \dots, W_V^{(g)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$

> **特殊情况**：当 $g = h$ 时，GQA 退化为标准的 **MHA**；当 $g = 1$ 时，GQA 退化为 **MQA**。通常大模型中会选择 $g = 8$。

### 4.2 详细计算步骤

GQA 的计算通过“分组共享”的逻辑进行，以下是具体的 4 个步骤：

#### 4.2.1 Step 1: 线性映射（Linear Projection）

将输入矩阵 $X$ 通过各自的线性层进行投影。

* 得到 $h$ 个 Query 矩阵：$Q_i = X W_Q^{(i)} \quad (i = 1, \dots, h)$
* 得到 $g$ 个 Key 矩阵：$K_j = X W_K^{(j)} \quad (j = 1, \dots, g)$
* 得到 $g$ 个 Value 矩阵：$V_j = X W_V^{(j)} \quad (j = 1, \dots, g)$

#### 4.2.2 Step 2: 映射 Query 到对应的 KV 组

每一个 Query 头 $Q_i$ 都有一个专属的 KV 组索引 $j$。对应关系为：

$$j = \lfloor (i - 1) / m \rfloor + 1$$

也就是说，前 $m$ 个 Query 头共享 $K_1, V_1$，接下来的 $m$ 个 Query 头共享 $K_2, V_2$，以此类推。

#### 4.2.3 Step 3: 独立计算每个头的注意力

每个 Query 头 $Q_i$ 与它所属组的 $K_j$ 和 $V_j$ 进行标准的缩放点积注意力计算：

$$\text{head}_i = \text{Attention}(Q_i, K_j, V_j) = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

* **维度**：$\text{head}_i \in \mathbb{R}^{n \times d_v}$

#### 4.2.4 Step 4: 拼接与最终映射（Concat & Output）

将所有 $h$ 个头的输出进行拼接，并通过最终的输出矩阵 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 融合信息：

$$\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

* **维度**：$(n \times d_{\text{model}})$

### 4.3 并行化矩阵维度的变化对照

在工程实现中，我们不会用循环来处理分组，而是利用张量重塑（Reshape）和 PyTorch 的广播（Broadcasting）来并行计算。

以下是三种 Attention 机制在并行化计算时的张量维度对比（假设 `batch_size` 维度已省略）：

| 机制 | Query ($Q$) 维度 | Key ($K$) & Value ($V$) 维度 | 广播与对齐方式 |
| --- | --- | --- | --- |
| **MHA** | `[h, n, d_k]` | `[h, n, d_k]` | 头数完全一致，1对1计算。 |
| **MQA** | `[h, n, d_k]` | `[1, n, d_k]` | KV 头部维度为 1，自动广播给所有 $h$ 个 Q 头。 |
| **GQA** | `[g, m, n, d_k]` | `[g, 1, n, d_k]` | 将 Query 拆出组维度 $g$ 和组内头维度 $m$；KV 拆出组维度 $g$ 且组内头维度为 1。在组内进行广播。 |

### 4.4 代码实现

```python 
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any

class GroupedQueryAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, n_groups: int) -> None:
        super(GroupedQueryAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_groups = n_groups
        self.group_size = self.n_heads // n_groups
        self.d_k = self.d_model // self.n_heads
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        # KV 的总维度缩小为: 分组数 * 单头维度
        self.W_k = nn.Linear(self.d_model, self.n_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.n_groups * self.d_k, bias=False)
        self.attention = ScaleDotProductAttention()
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None) -> Tensor:
        batch_size = Q.shape[0]
        # 步骤 1：映射成投影矩阵
        q_proj = self.W_q(Q)  # [B, L_q, d_model]
        k_proj = self.W_k(K)  # [B, L_k, num_groups * d_k]
        v_proj = self.W_v(V)  # [B, L_v, num_groups * d_k]
        # 步骤 2：调整维度，引入组(Group)概念
        # Q 拆分为 5D 张量: [B, L_q, num_groups, group_size, d_k] -> 
        # 转置为 [B, num_groups, group_size, L_q, d_k]
        q_heads = q_proj.view(batch_size, -1, self.n_groups, self.group_size, self.d_k).permute(0, 2, 3, 1, 4)
        # K 和 V 同样拆为 5D，但组内头数设为 1: [B, L_k, num_groups, 1, d_k] -> 
        # 转置为 [B, num_groups, 1, L_k, d_k]
        k_heads = k_proj.view(batch_size, -1, self.n_groups, 1, self.d_k).permute(0, 2, 3, 1, 4)
        v_heads = v_proj.view(batch_size, -1, self.n_groups, 1, self.d_k).permute(0, 2, 3, 1, 4)
        # 在内部自动转换 Mask 的维度
        if mask is not None:
            # 情况 A：如果传入的是纯 2D 掩码 [len_q, len_k] (例如全局共享的因果掩码)
            if mask.dim() == 2:
                # [len_q, len_k] -> [1, 1, 1, len_q, len_k]
                mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            # 情况 B：如果传入的是标准 3D 批处理掩码 [B, len_q, len_k] (例如带 Padding 的掩码)
            elif mask.dim() == 3:
                # [B, len_q, len_k] -> [B, 1, 1, len_q, len_k]
                mask = mask.unsqueeze(1).unsqueeze(2)
        # 步骤 3：调用 ScaledDotProductAttention
        # q_heads: [B, G, M, L_q, d_k]
        # k_heads: [B, G, 1, L_k, d_k] -> matmul 在第 2 维(M与1)自动广播
        context = self.attention(q_heads, k_heads, v_heads, mask=mask)
        # context 维度: [B, num_groups, group_size, L_q, d_k]
        # 步骤 4：恢复并还原维度 (将组和组内头数合并回原先的多头)
        # 先换回序列长度在前: [B, L_q, num_groups, group_size, d_k]
        context = context.permute(0, 3, 1, 2, 4).contiguous()
        # 展平回 3D 张量: [B, L_q, d_model]
        output = context.view(batch_size, -1, self.d_model)
        # 步骤 5：输出线性映射
        output = self.W_O(output)
        return output
```

## 5. 总结

本文综述了 Transformer 注意力机制的演进历程。从最初的 Scaled Dot-Product Attention 到 Multi-Head Attention，再到针对推理优化的 Multi-Query Attention 和折中方案 Grouped-Query Attention，研究者们始终在模型表达能力与推理效率之间寻求平衡。

| 机制 | 论文来源 | KV 头数 | 推理效率 | 表达能力 |
| --- | --- | --- | --- | --- |
| **MHA** | Vaswani et al., 2017 | $h$ 个 | 较低 | 强 |
| **MQA** | Shazeer et al., 2019 | 1 个 | 高 | 弱 |
| **GQA** | Ainslie et al., 2023 | $g$ 个 | 中等 | 中等 |

综上所述，GQA 作为 MHA 与 MQA 的折中方案，在保持较好表达能力的同时显著降低了 KV Cache 开销，已成为当代大语言模型的主流选择。


