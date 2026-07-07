# 位置编码（Positional Encoding）

位置编码（Positional Encoding）是 Transformer 架构中的关键组件之一。由于 Transformer 采用自注意力机制进行信息交互，其本身不具备处理序列顺序的能力——无论输入序列如何打乱，注意力计算的输出结果都是相同的。位置编码的引入正是为了解决这一问题，使模型能够区分序列中不同位置的 token。

本文围绕 Transformer 中的位置编码技术展开综述，介绍绝对位置编码与相对位置编码两类主流方法。

## 1. 背景与问题定义

在 Transformer 的自注意力计算中：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

公式本身对输入序列的顺序是**对称的**——交换输入序列中两个 token 的位置，注意力矩阵的值不会发生任何变化。这意味着，如果不额外引入位置信息，模型将无法区分"我爱你"和"你爱我"这类语序敏感的任务。

位置编码的核心目标是为序列中的每个位置 $i$ 生成一个独特的向量 $PE_i$，将其与 token 的嵌入向量相加，从而使模型能够感知位置信息。

---

绝对位置编码是最早被提出的位置编码方案，为序列中的每个绝对位置生成一个固定的编码向量。

## 2. Learnable 位置编码

Learnable 位置编码（可学习位置编码）最早由 Bahdanau 等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》（2014）中提出，随后被广泛应用于早期神经机器翻译模型。

其核心思想是将位置编码视为可学习的参数矩阵 $PE \in \mathbb{R}^{L \times d_{\text{model}}}$，通过反向传播自动学习最优的位置表示。

$$\text{Input}_i = \text{TokenEmbedding}_i + \text{PositionEmbedding}_i$$

### 2.1 代码实现

```python
class LearnablePositionalEncoding(torch.nn.Module):
    """
    Learnable Positional Encoding
    论文来源：Bahdanau et al., 2014
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor, shape = [batch_size, seq_len, d_model]
        :return: Tensor, shape = [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        return x + self.position_embeddings(positions)
```

## 3. Sinusoidal 位置编码

Sinusoidal 位置编码（正弦位置编码）是 Transformer 架构奠基之作 *Attention Is All You Need* 中提出的一种**绝对位置编码**技术。它不需要通过数据训练去学习，而是直接利用不同频率的正弦和余弦函数组合，为序列中的每个位置生成一个固定的、唯一的特征向量。

### 3.1 核心数学公式

假设我们需要为一个长度为 $L$ 的文本序列生成位置编码，每个位置的位置编码向量维度为 $d$（通常与 Embedding 维度相同，且为偶数）。

对于序列中的第 $pos$ 个位置（$pos \in [0, L-1]$），以及该位置编码向量中的第 $i$ 个维度（$i \in [0, d-1]$），其计算公式定义如下：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

#### 公式拆解：

* **$pos$**：当前 Token 在句子中的绝对位置（如第 0 个词、第 1 个词...）。
* **$2i$ / $2i+1$**：代表位置编码向量中的索引。**偶数维度使用正弦（$\sin$），奇数维度使用余弦（$\cos$）**。
* **$10000^{\frac{2i}{d}}$**：这是一个分母，通常定义为波长控制参数。随着维度索引 $i$ 的增大，分母呈指数级变大，这意味着**波长越来越长，频率越来越低**。

### 3.2 完整的计算步骤（矩阵视角）

在工程实现中，通常不会用双重循环去逐个元素计算，而是采用矩阵化操作。以下是完整的全矩阵计算流程：

**第一步：构建位置向量 (Position Vector)**

创建一个形状为 $(L, 1)$ 的列向量，代表序列中每个点的绝对位置：

$$pos = \begin{bmatrix} 0 \\ 1 \\ 2 \\ \vdots \\ L-1 \end{bmatrix}$$

**第二步：计算逆频率向量 (Inverse Frequency Vector)**

根据维度的前半部分（偶数项），计算缩放系数（也叫角速度 $\omega_i$）。共有 $d/2$ 个项：

$$\text{inv_freq} = \left[ \frac{1}{10000^0}, \frac{1}{10000^{\frac{2}{d}}}, \frac{1}{10000^{\frac{4}{d}}}, \dots, \frac{1}{10000^{\frac{d-2}{d}}} \right]$$

其形状为 $(1, d/2)$ 的行向量。

**第三步：计算角度矩阵 (Angles Matrix)**

将位置列向量与逆频率行向量进行外积（Matrix Outer Product）相乘，得到一个形状为 $(L, d/2)$ 的角度矩阵。矩阵中的每一个元素都是 $pos \cdot \omega_i$：

$$\text{angles} = pos \times \text{inv_freq} = \begin{bmatrix}
0 \cdot \omega_0 & 0 \cdot \omega_1 & \cdots & 0 \cdot \omega_{d/2-1} \\
1 \cdot \omega_0 & 1 \cdot \omega_1 & \cdots & 1 \cdot \omega_{d/2-1} \\
\vdots & \vdots & \ddots & \vdots \\
(L-1) \cdot \omega_0 & (L-1) \cdot \omega_1 & \cdots & (L-1) \cdot \omega_{d/2-1}
\end{bmatrix}$$

**第四步：应用 $\sin$ 和 $\cos$ 并拼接**

1. 对角度矩阵的每一个元素求正弦，得到 $\sin(\text{angles})$，形状为 $(L, d/2)$。
2. 对角度矩阵的每一个元素求余弦，得到 $\cos(\text{angles})$，形状为 $(L, d/2)$。

最后，将这两个矩阵横向拼接。在标准实现中，通常是以**交错（Interleave）**或者**前后拼接**的方式组合。

如果采用**交错拼接**，最终得到的位置编码矩阵 $PE$ 形状为 $(L, d)$：

$$PE = \begin{bmatrix}
\sin(0\cdot\omega_0) & \cos(0\cdot\omega_0) & \sin(0\cdot\omega_1) & \cos(0\cdot\omega_1) & \cdots \\
\sin(1\cdot\omega_0) & \cos(1\cdot\omega_0) & \sin(1\cdot\omega_1) & \cos(1\cdot\omega_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

### 3.3 直观理解：它长什么样？

如果我们将最终的 $PE$ 矩阵可视化为一张热力图：

* **纵轴（位置 $pos$）**：从上往下代表句子的第一个词到最后一个词。
* **横轴（维度 $d$）**：从左往右。
* **左侧（低维区域）**：$\sin$ 和 $\cos$ 的频率非常高，图形呈现密集的正弦波浪，能够精确捕捉**近距离、微观**的位置差异。
* **右侧（高维区域）**：频率极低，图形随着 $pos$ 的增加变化非常缓慢，甚至趋于直线，用于捕捉**长距离、宏观**的全局位置框架。

这种设计使得每一个位置的 $d$ 维向量都是独一无二的"数字指纹"。

### 3.4 为什么要这样设计？（设计巧妙之处）

原论文指出，选择正弦和余弦函数的核心原因在于：**它允许模型轻松学习到相对位置关系。**

根据三角函数的和差化积公式：

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$

$$\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$

这意味着，对于任何固定的相对偏移量 $k$，位置 $pos+k$ 的位置编码向量 $PE_{pos+k}$，都可以表示为位置 $pos$ 的位置编码向量 $PE_{pos}$ 的**线性变换**。

也就是说，Attention 机制在计算两个相距为 $k$ 的词的注意力时，它们之间的点积可以通过线性矩阵变换，转化成只与相对距离 $k$ 相关的变换，从而让模型既具备绝对位置感，又具备动态的相对距离感。

最后，生成好这个 $(L, d)$ 的 $PE$ 矩阵后，在 Transformer 中直接将其与原始词向量矩阵（Token Embeddings）进行**按位置相加（Element-wise Add）**，便完成了位置信息的注入：

$$\text{Input to Transformer} = \text{Token Embeddings} + PE$$

### 3.5 代码实现

```python
import torch
import math

class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Sinusoidal Positional Encoding
    论文来源：Vaswani et al., 2017
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # 计算除数 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 便于批量相加
        self.register_buffer('pe', pe)  # 注册为 buffer，不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor, shape = [batch_size, seq_len, d_model]
        :return: Tensor, shape = [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]
```

### 3.6 绝对位置编码对比

| 特性 | Learnable | Sinusoidal |
|------|-----------|------------|
| 论文来源 | Bahdanau et al., 2014 | Vaswani et al., 2017 |
| 参数量 | $L \times d_{\text{model}}$ | 无需额外参数 |
| 外推能力 | 有限（受 max_len 限制） | 理论支持（周期函数） |
| 灵活性 | 可学习、更灵活 | 固定、不可调优 |
| 效果 | 通常略优于 Sinusoidal | 表现稳定 |

---

绝对位置编码在处理长序列时存在明显的局限性：一方面，模型难以直接学习位置之间的相对关系；另一方面，当测试序列长度超过训练集最大长度时，编码效果急剧下降。相对位置编码通过直接编码 token 之间的相对距离来解决这些问题。

## 4. RoPE 位置编码

旋转位置编码（Rotary Position Embedding，RoPE）由 Su 等人在论文《RoFormer: Enhanced Transformer with Rotary Position Embedding》（2022）中提出，是目前大语言模型（如 LLaMA、Baichuan、Mistral 等）中最主流的位置编码技术之一。它的核心思想是：通过在复数空间中旋转 Query 和 Key 向量，将绝对位置信息自然地转化为相对位置信息。

### 4.1 核心数学思想

RoPE 的核心目标是：**通过一个变换函数 $f(x, \text{pos})$，将位置信息编码到向量中，使得变换后的向量进行点积运算时，结果只依赖于相对位置。**

具体来说，假设输入两个特征向量 $x_q$ 和 $x_k$，分别处于绝对位置 $m$ 和 $n$。我们希望找到一个变换函数 $f(x, \text{pos})$，满足：

$$\langle f(x_q, m), f(x_k, n) \rangle = g(x_q, x_k, m - n)$$

RoPE 巧妙地利用了**二维空间中的旋转矩阵**来实现这一特性。在二维空间中，旋转一个向量不改变其长度，只改变其角度。

在 RoPE 中，这个变换函数的具体形式为：

$$f(x, \text{pos}) = R_{\Theta, \text{pos}}^d \cdot x$$

其中 $R_{\Theta, \text{pos}}^d$ 是一个分块对角旋转矩阵，将位置 $\text{pos}$ 编码为旋转角度。

### 4.2 完整的计算步骤

在实际的 Transformer 架构中，假设多头注意力机制中某个头的维度为 $d$（$d$ 通常为偶数，如 64 或 128）。RoPE 的具体计算流程如下：

**第一步：拆分二维子空间**

因为旋转是在二维平面上定义的，RoPE 会将一个 $d$ 维的 Query（或 Key）向量拆分为 $d/2$ 个独立的**二维子空间**。

对于位置为 $m$ 的 $d$ 维向量 $q_m = [q_0, q_1, q_2, q_3, \dots, q_{d-2}, q_{d-1}]^T$，拆分为：

$$\begin{bmatrix} q_0 \\ q_1 \end{bmatrix}, \begin{bmatrix} q_2 \\ q_3 \end{bmatrix}, \dots, \begin{bmatrix} q_{d-2} \\ q_{d-1} \end{bmatrix}$$

**第二步：计算每个子空间的旋转角度 $\theta_i$**

RoPE 为不同的二维子空间分配不同的旋转基数（类似于 Transformer 原始正弦位置编码）。

对于第 $i$ 个二维子空间（$i \in [0, 1, \dots, \frac{d}{2}-1]$），其旋转角速度 $\theta_i$ 计算公式为：

$$\theta_i = \text{base}^{-\frac{2i}{d}} \quad (\text{通常 } \text{base} = 10000)$$

在位置 $m$ 处，该子空间实际旋转的角度为：

$$\omega_i = m \theta_i$$

**第三步：应用旋转变换（矩阵相乘）**

对于第 $i$ 个二维子空间 $\begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$，乘以其对应的二维旋转矩阵 $R_m^{(i)}$：

$$\begin{bmatrix} \tilde{q}_{2i} \\ \tilde{q}_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

展开后的计算结果为：

- $\bar q_{2i} = q_{2i} \text{cos}(m\theta_i) - q_{2i+1} \text{sin}(m\theta_i)$
- $\bar q_{2i+1} = q_{2i} \sin(m\theta_i) + q_{2i+1} \cos(m\theta_i)$

将所有子空间旋转后的结果重新拼接，就得到了带有位置 $m$ 信息的全维度向量 $\tilde{q}_m$。

### 4.3 整体矩阵表示

如果将上述过程写成一个整体的大矩阵乘以 $q_m$，RoPE 的全局变换矩阵 $R_{\Theta, m}^d$ 是一个分块对角矩阵：

$$\tilde{q}_m = R_{\Theta, m}^d q_m = \begin{bmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{\frac{d}{2}-1} & -\sin m\theta_{\frac{d}{2}-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{\frac{d}{2}-1} & \cos m\theta_{\frac{d}{2}-1}
\end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{bmatrix}$$

### 4.4 高效的工程实现

在代码实现中，直接进行大型稀疏矩阵乘法效率极低。为了加速，通常会利用逐元素相乘（Element-wise Product）的 trick。

我们将原向量 $q_m$ 复制并做一个"交错取反"的操作，定义一个辅助向量 $q_m^{\text{half}}$：

$$q_m^{\text{half}} = [-q_1, q_0, -q_3, q_2, \dots, -q_{d-1}, q_{d-2}]^T$$

同时，我们将所有的 $\cos(m\theta_i)$ 和 $\sin(m\theta_i)$ 复制扩展成与 $q$ 维度相同的 $d$ 维向量：

* $C_m = [\cos(m\theta_0), \cos(m\theta_0), \cos(m\theta_1), \cos(m\theta_1), \dots]^T$
* $S_m = [\sin(m\theta_0), \sin(m\theta_0), \sin(m\theta_1), \sin(m\theta_1), \dots]^T$

最终的 RoPE 计算可以精简为两组向量的哈达玛积（Hadamard Product，即对应位置直接相乘）之和：

$$\tilde{q}_m = q_m \odot C_m + q_m^{\text{half}} \odot S_m$$

> **提示：** 在很多主流模型（如 LLaMA）的代码实现中，为了内存对齐和计算方便，二维切分的方式不是 $[0,1], [2,3]$ 邻近切分，而是**前后对半切分**（前 $\frac{d}{2}$ 维与后 $\frac{d}{2}$ 维两两配对）。虽然切分方式不同，但数学本质完全一致。

### 4.5 为什么能表示相对位置？

当计算位置 $m$ 的 Query 和位置 $n$ 的 Key 的点积时：

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = (R_{\Theta, m}^d q_m)^T (R_{\Theta, n}^d k_n) = q_m^T (R_{\Theta, m}^d)^T R_{\Theta, n}^d k_n$$

由于旋转矩阵是**正交矩阵**，满足 $(R_m)^T = R_{-m}$，且矩阵乘法满足角度相加减：

$$(R_{\Theta, m}^d)^T R_{\Theta, n}^d = R_{\Theta, -m}^d R_{\Theta, n}^d = R_{\Theta, n - m}^d$$

因此：

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = q_m^T R_{\Theta, n - m}^d k_n$$

**结论：** 经过 RoPE 编码后，Query 和 Key 的点积结果只取决于它们的相对距离 $n - m$（或 $m - n$），这完美实现了相对位置编码的效果，同时保留了绝对位置的线性变换能力。

### 4.6 代码实现

```python
import torch
import math

class RoPEPositionalEncoding(torch.nn.Module):
    """
    Rotary Position Embedding
    论文来源：Su et al., 2022
    """
    def __init__(self, dim: int, base: int = 10000):
        super(RoPEPositionalEncoding, self).__init__()
        self.dim = dim
        self.base = base
        # 预计算旋转角度 inv_freq[i] = theta_i
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device = None):
        """
        生成旋转角度矩阵
        :param seq_len: 序列长度
        :param device: 计算设备
        :return: 旋转角度矩阵 [seq_len, dim/2]
        """
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # 外积得到 [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return emb

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        应用旋转编码到输入向量
        :param x: 输入向量 [batch, num_heads, seq_len, dim]
        :param cos: 余弦值 [seq_len, dim]
        :param sin: 正弦值 [seq_len, dim]
        :return: 旋转后的向量
        """
        # 将 cos/sin 扩展为与 x 相同的维度
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        # 旋转公式：x_rot = x * cos + rotate(x) * sin
        # 其中 rotate(x) = [-x[..., 1::2], x[..., ::2]]
        x1 = x[..., :x.size(-1)//2]
        x2 = x[..., x.size(-1)//2:]
        return torch.cat([
            x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x2.size(-1)],
            x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x2.size(-1)]
        ], dim=-1)
```

## 5. YaRN 位置编码

YaRN（Yet another RoPE extensioN method）是目前大语言模型（LLM）用于**长文本上下文扩展（Context Window Extension）**最先进、效果最好的技术之一。

它的核心痛点在于：标准的 RoPE 无法直接推广到比预训练更长的序列（会外推失败）。YaRN 基于"NTK-by-parts"插值法，通过对**不同特征维度（频率）进行分段非线性缩放**，并引入**温度修正**，使得模型只需极少的数据微调，就能将上下文窗口扩展 16 倍以上（例如从 4k 扩展到 64k/128k）。

### 5.1 核心理论：频率分段（NTK-by-parts）

RoPE 的本质是不同维度有着不同的旋转"波长"。YaRN 的核心观察是：**不同波长的维度，在扩展上下文时扮演的角色完全不同。**

若预训练的最大长度为 $L$：

1. **高频维度（波长 $\lambda < L$）**：这些维度在预训练的窗口内已经旋转了不止一圈，它们主要编码**局部绝对位置和紧密相对关系**。如果盲目对它们进行压缩（插值），会破坏原有的局部感知力。因此，**YaRN 对高频区域不压缩（保留外推）**。

2. **低频维度（波长 $\lambda > L$）**：这些维度在整个预训练窗口内连一圈都没转完，其点积不具备跨越全周期的相对位置区分度，主要编码**全局宏观框架**。如果直接外推，模型会遇到未见过的角度组合导致崩塌。因此，**YaRN 对低频区域进行完整的线性压缩（插值）**。

3. **中频维度**：处于两者之间，**YaRN 采用平滑过渡（Smooth Blend）**。

### 5.2 完整的计算步骤

假设我们要将上下文从原长度 $L$ 扩展到新长度 $L' = s \cdot L$（其中 $s$ 为扩展倍率，如 $s=4$ 或 $s=16$）。

**第一步：计算原始的波长 $\lambda_i$**

对于特征维度为 $d$ 的每个二维子空间索引 $i \in [0, 1, \dots, \frac{d}{2}-1]$，原始的角速度和波长分别为：

$$\theta_i = \text{base}^{-\frac{2i}{d}}$$

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \text{base}^{\frac{2i}{d}}$$

**第二步：定义分段边界（超参数 $\alpha$ 和 $\beta$）**

YaRN 引入了两个超参数 $\alpha$ 和 $\beta$ 来界定频率：

* 当 $\lambda_i < \alpha L$ 时，判定为高频（纯外推）。
* 当 $\lambda_i > \beta L$ 时，判定为低频（纯插值）。

为了方便计算，通常将其转化为**维度索引边界**（即根据上面的波长公式反推出具体的维度 $i$）：

* **高频边界对应的维度索引 $r_{\text{fast}}$**：

$$r_{\text{fast}} = \frac{d}{2} \cdot \frac{\ln(\alpha L / 2\pi)}{\ln(\text{base})}$$

* **低频边界对应的维度索引 $r_{\text{slow}}$**：

$$r_{\text{slow}} = \frac{d}{2} \cdot \frac{\ln(\beta L / 2\pi)}{\ln(\text{base})}$$

**第三步：计算渐变比例函数 $\gamma_i$**

对于当前的维度索引 $i$，引入一个斜坡函数（Ramp Function）$\gamma_i$，用于在插值与外推之间做平滑过渡：

$$\gamma_i = \begin{cases}
0 & \text{if } i < r_{\text{fast}} \quad (\text{高频，完全外推}) \\
1 & \text{if } i > r_{\text{slow}} \quad (\text{低频，完全插值}) \\
\frac{i - r_{\text{fast}}}{r_{\text{slow}} - r_{\text{fast}}} & \text{otherwise} \quad (\text{中频，线性渐变})
\end{cases}$$

**第四步：计算 YaRN 的新频率（角速度）$\theta'_i$**

利用 $\gamma_i$，将原始频率 $\theta_i$ 与 线性缩放后的频率 $\frac{\theta_i}{s}$ 进行加权混合。YaRN 改进的实际修正频率公式为：

$$\theta'_i = (1 - \gamma_i) \cdot \theta_i + \gamma_i \cdot \frac{\theta_i}{s}$$

* **解读：** 当 $\gamma_i=0$ 时，$\theta'_i = \theta_i$（保留原样不压缩）；当 $\gamma_i=1$ 时，$\theta'_i = \theta_i / s$（频率变慢，完美容纳 $s$ 倍长文本）。

**第五步：注意力温度修正（Attention Temperature Scaling）**

这是 YaRN 相比之前方法的另一大关键创新。当位置编码被压缩后，不同位置向量之间的平均点积值会变小（因为角度靠得更近了）。这会导致 Attention 矩阵的熵增加，分布变得极其平缓，从而丧失了注意力集中的能力。

为了抵消这一副作用，YaRN 引入了一个注意力温度放大系数 $t$：

$$t = 0.1 + \frac{1.16 \cdot \ln(s)}{\ln(10)}$$

在计算应用完位置编码的 Query ($q$) 和 Key ($k$) 的点积时，通过乘以系数 $\sqrt{\frac{1}{t}}$（或者直接将对应的 $q, k$ 向量除以 $\sqrt{t}$）来放大点积差异，强行恢复注意力矩阵的动态范围：

$$\text{Attention Score} = \text{Softmax}\left( \frac{1}{t} \cdot \frac{q^T k}{\sqrt{d_{\text{head}}}} \right)$$

### 5.3 最终矩阵应用

最终，对于任何一个绝对位置为 $m$ 的 Token，其对应的第 $i$ 个二维子空间，所采用的旋转矩阵不再是标准的 $m\theta_i$，而是替换为了：

$$\begin{bmatrix} \tilde{q}_{2i} \\ \tilde{q}_{2i+1} \end{bmatrix} = \frac{1}{\sqrt{t}} \begin{bmatrix} \cos(m\theta'_i) & -\sin(m\theta'_i) \\ \sin(m\theta'_i) & \cos(m\theta'_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

### 5.4 总结与意义

YaRN 位置编码计算流程的伟大之处在于：

* **局部高精度**：保证了相邻 Token 之间的短距离高频相对位置信息完全不失真。
* **全局自适应**：长距离的信息通过动态插值被优雅地压缩在新窗口内。
* **无痛扩展**：通过温度因子 $t$ 解决了点积分布涣散的工程难题。

这也使得 LLaMA-2/3、Mistral 等主流开源模型在需要扩展长文本时，普遍首选基于 YaRN 变体的解决方案。

### 5.5 代码实现

```python
import torch
import math

class YaRNPositionalEncoding(torch.nn.Module):
    """
    Yet another RoPE extensioN
    论文来源：Peng et al., 2023
    """
    def __init__(self, dim: int, base: int = 10000, original_max_position_embeddings: int = 2048,
                 scaling_factor: float = 1.0):
        super(YaRNPositionalEncoding, self).__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        # 原始训练的最大长度
        self.original_max_position_embeddings = original_max_position_embeddings
        # 重新计算缩放后的 base
        self.base = base * (scaling_factor ** (dim / (dim - 2)))
        # 预计算旋转角度
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device = None):
        """生成旋转角度矩阵（支持缩放）"""
        if device is None:
            device = self.inv_freq.device
        # 对位置索引进行缩放
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor  # 线性缩放
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """应用旋转编码（与 RoPE 相同）"""
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., :x.size(-1)//2]
        x2 = x[..., x.size(-1)//2:]
        return torch.cat([
            x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x2.size(-1)],
            x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x2.size(-1)]
        ], dim=-1)
```

## 6. 总结

本文综述了位置编码技术的发展历程，从最早的可学习位置编码（Bahdanau et al., 2014），到 Transformer 的标准方案 Sinusoidal（Vaswani et al., 2017），再到近年来针对长序列优化的相对位置编码（RoPE、YaRN），研究者们始终在解决一个核心问题：如何让模型有效感知序列中的位置信息。

| 方法 | 论文来源 | 类型 | 外推能力 |
|------|----------|------|----------|
| Learnable | Bahdanau et al., 2014 | 绝对 | 有限 |
| Sinusoidal | Vaswani et al., 2017 | 绝对 | 理论支持 |
| RoPE | Su et al., 2022 | 相对 | 强 |
| YaRN | Peng et al., 2023 | 相对 | 最强 |

当前主流大语言模型普遍采用 RoPE 或 YaRN 作为位置编码方案。YaRN 通过 NTK 感知缩放技术，在长文本处理方面展现出显著优势，已成为大模型长上下文扩展的推荐方案。
