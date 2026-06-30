# Attention 

## 1. Scaled Dot-Product Attention

输入序列包含$n$个向量,每个向量的特征维度为$d_{\text{model}}$
首先输入矩阵$X \in \mathbb{R}^{n \times d_{\text{model}}$会通过三个不同的线性变换矩阵(权重矩阵),映射得到三个核心矩阵

- Query(查询矩阵) Q: $Q = XW_Q$ (其中$W_Q \in \mathbb{R}^{d_{\text{model} \times d_k}$)
- Key(键矩阵) K: $K = XW_K$ (其中$W_K \in \mathbb{R}^{d_{\text{model} \times d_k}$)
- Value(值矩阵) V: $V = XW_V$ (其中$W_V \in \mathbb{R}^{d_{\text{model} \times d_v}$)

注: $d_k = d_v = d_{\text{model} / \text{num\_heads}}$

计算步骤:

STEP 1: 计算原始注意力分数(Scores)

通过将$Q$和$K$的转置进行矩阵乘法,计算出Query和Key之间的相似度，点积结果越大，说明两个向量的相关性越高。

$$\text{Scores} = Q K^T$$

- 维度变化: $(n \times d_k) \times (d_k \times n) \rightarrow (n \times n)$
- 结果矩阵中的第 $(i, j)$ 个元素，代表第 $i$ 个单词对第 $j$ 个单词的原始注意力得分。

STEP 2: 缩放操作(Scaling)

将原始分数除以$\sqrt{d_k}$,即Key向量维度的平方根

$$\text{Scaled Scores} = \frac{Q K^T}{\sqrt{d_k}}$$

- 目的: 当 $d_k$ 很大时，点积的结果会非常大，导致后面经过 Softmax 函数时梯度变得极小（进入饱和区）。除以 $\sqrt{d_k}$ 可以起到方差缩放的作用，让训练更加稳定。

STEP 3: 归一化(Softmax)

对缩放后的分数在**行方向**上应用 Softmax 函数，将其转化为概率分布（所有权重相加为 1）。

$$\text{Attention Weights} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

- **维度**：$(n \times n)$
- 此时矩阵里的数值就是最终的**注意力权重**（矩阵中每行的和为 1）

STEP 4: 加权求和（Output）

用计算出的注意力权重矩阵去乘以 Value 矩阵 $V$，实现对 Value 的加权聚合。

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **维度变化**：$(n \times n) \times (n \times d_v) \rightarrow (n \times d_v)$
- 最终输出的矩阵中，每一个向量都融合了整个序列中与其相关的其他向量的信息。
