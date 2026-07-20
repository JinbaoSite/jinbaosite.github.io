# RankMixer：工业级推荐排序模型的规模化之路

> 论文：*RankMixer: Scaling Up Ranking Models in Industrial Recommenders*
> 机构：字节跳动（抖音推荐算法团队）
> arXiv: 2507.15551


## 1 动机

LLM 通过不断堆参数验证了 Scaling Law，一个自然的问题是：**推荐排序模型能不能也这样做大？** 现实中有两个卡点：

1. **硬约束**：线上精排要在十几毫秒内完成打分并支撑高 QPS，不可能像训练 LLM 那样无限堆算力。
2. **架构落后**：现有排序模型（DeepFM、DCN、AutoInt、DHEN 等）依赖内积、逐元素乘法、注意力等人工特征交叉算子，这些算子大多是**访存密集型（Memory-bound）**而非**计算密集型（Compute-bound）**，无法吃满现代 GPU 的大矩阵乘算力——现代 LLM 的 MFU（算力利用率）普遍在 **40%+**，而字节线上精排模型当时只有 **4.5%**。MFU 低意味着参数量和计算成本近似线性绑定，一旦放大参数，FLOPs 和时延立刻爆炸，Scaling Law 的收益还没吃到，成本就先扛不住了。

为此，论文提出 **RankMixer**：一种硬件感知（hardware-aware）的统一特征交互架构，同时满足"硬件对齐（转化为 GPU 高效的大矩阵乘）"与"契合推荐数据特性（建模数百个异构特征域的个性化交互）"两个约束。核心做法是**用无参数的 Multi-Head Token Mixing 替代平方复杂度的 Self-Attention**，用 **Per-token FFN** 建模特征子空间的独立表达与交互，并扩展出 **Sparse-MoE 变体**进一步提升容量。

最终效果：MFU 从 **4.5% 提升到 45%**（约10倍），在**几乎不增加推理时延**的前提下将线上模型参数规模从 16M 提升到 **1.1B（约70倍）**，全量上线抖音 Feed 精排后用户活跃天数 **+0.3%**、App 使用时长 **+1.08%**。


## 2 模型架构

RankMixer 整体架构类似 Transformer 的层次堆叠结构：输入被 Token 化为 $T$ 个特征 Token，经过 $L$ 层 RankMixer Block 逐层refine，最后做 mean pooling 得到输出表征，用于多任务预测（如 finish、skip、like、comment 等）。

### 2.1 总体公式

每一个 RankMixer Block 包含两个核心模块：**Multi-Head Token Mixing** 和 **Per-Token FFN（PFFN）**，公式为：

$$
S_{n-1} = \mathrm{LN}\big(\mathrm{TokenMixing}(X_{n-1}) + X_{n-1}\big)
$$

$$
X_n = \mathrm{LN}\big(\mathrm{PFFN}(S_{n-1}) + S_{n-1}\big)
$$

其中：
- $\mathrm{LN}(\cdot)$ 是 LayerNorm
- $X_n \in \mathbb{R}^{T \times D}$ 是第 $n$ 层 RankMixer Block 的输出
- $X_0 \in \mathbb{R}^{T \times D}$ 由初始的 $T$ 个 Token（$x_1, x_2, ..., x_T$）堆叠而成
- $D$ 是模型隐藏维度

最终输出 $o_{output}$ 来自最后一层 $X_L$ 的 mean pooling，再接不同任务头输出预测。

整体结构如下图所示（对应论文 Figure 1，图中以 T=4 个 Token、Multi-Head Token Mixing 拆出 3 个 head 为例展示了内部机制）：

![RankMixer架构图](https://raw.githubusercontent.com/JinbaoSite/jinbaosite.github.io/master/img/RankMixer.svg)

*图1：RankMixer 整体架构。输入特征经 Tokenization 得到 T 个 Token，堆叠 L 层 RankMixer Block（Multi-Head Token Mixing + Per-token FFN/SMoE，均带残差与LayerNorm），最后 mean pooling 输出多任务预测。*

### 2.2 输入层与特征 Tokenization

#### 2.2.1 为什么需要 Tokenization

推荐模型的输入特征包括：

- **User Profiles**：用户 ID、用户画像等
- **Video/Candidate Features**：视频 ID、作者 ID 等
- **Sequence Features**：经过序列建模模块（如 LONGER）处理后的用户行为序列表征 $e_s$
- **Cross Features（Interacted Features）**：用户与候选物品之间的交叉特征

这些特征会先各自 Embedding 化，得到维度不一的 embedding 向量。为了在后续阶段实现高效并行计算，必须把这些"维度参差不齐"的 embedding 转换为**维度对齐的向量**，论文称这一过程为 **Tokenization**。

论文分析了两种朴素策略的问题：

1. **每个特征一个 Token**：由于特征数量高达数百个，会导致每个 Token 分到的参数量和计算量都极小，重要特征建模不充分，同时大量小 Token 也会造成 GPU 利用率低下（矩阵形状太"瘦"，无法发挥 GEMM 优势）。
2. **只用一个 Token（全部特征拼接后过一个 DNN）**：模型退化为普通 DNN，无法区分不同特征子空间，高频/头部特征会淹没长尾特征信号。

#### 2.2.2 基于语义分组的 Tokenization 方案

RankMixer 采用**基于领域知识的语义分组策略**：先利用业务先验知识，将数百个特征按语义划分为 $N$ 个组（比如"用户画像组""视频内容组""交互统计组"等），组内特征顺序拼接得到一个大向量：

$$
e_{input} = [e_1; e_2; \dots; e_N]
$$

再将这个拼接向量**等距切分**为 $T$ 个固定维度 $d$ 的片段，每个片段经过投影映射到统一的模型维度 $D$，得到第 $i$ 个 Token：

$$
x_i = \mathrm{Proj}\big(e_{input}[d\cdot(i-1) : d\cdot i]\big), \quad i = 1, \dots, T
$$

其中：
- $e_{input}$：拼接后的特征向量
- $d$：每个 Token 切分前的固定维度
- $N$：特征分组数
- $T$：最终生成的 Token 数量
- $\mathrm{Proj}(\cdot)$：将切分片段映射到模型宽度 $D$ 的线性投影

这样每个 Token $x_i \in \mathbb{R}^D$ 都代表一组语义相对一致的特征子空间，既避免了 Token 过多导致的碎片化，也避免了单一 Token 导致的特征淹没问题。

### 2.3 RankMixer Block 详解

#### 2.3.1 Multi-Head Token Mixing：无参数的特征交互算子

这是 RankMixer 替代 Self-Attention 的核心模块，目的是让不同 Token（即不同特征子空间）之间做全局信息交换。

**具体做法**：每个 Token $x_t$ 被均分为 $H$ 个 head：

$$
\big[x_t^{(1)} \Vert x_t^{(2)} \Vert \dots \Vert x_t^{(H)}\big] = \mathrm{SplitHead}(x_t)
$$

可以把每个 head 理解成 Token 在某个低维子空间上的投影——因为推荐任务本身需要从多个不同"视角"看待特征。

然后，Token Mixing 做的事情是：**把所有 Token 的同一个 head 位置拼接（Concat）在一起，形成新的"混合 Token"**：

$$
s^h = \mathrm{Concat}\big(x_1^h, x_2^h, \dots, x_T^h\big), \quad h = 1, \dots, H
$$

即：第 $h$ 个混合 Token $s^h$，由原来 $T$ 个 Token 各自的第 $h$ 个 head 拼接而成。这本质上是一次**跨 Token 的维度重排（Shuffle）**，不引入任何可学习参数——因此是"parameter-free"的。

Token Mixing 输出为 $S \in \mathbb{R}^{H \times \frac{TD}{H}}$，由 $s^1, s^2, \dots, s^H$ 堆叠而成。论文中设定 **$H = T$**，即 head 数等于 Token 数，这样混合后 Token 数量保持不变，方便做残差连接：

$$
s_1, \dots, s_T = \mathrm{LN}\big(\mathrm{TokenMixing}(x_1,\dots,x_T) + (x_1,\dots,x_T)\big)
$$

**为什么不用 Self-Attention？** 论文给出了明确论证：Self-Attention 的注意力权重基于 Token 间的内积相似度计算，这在 NLP 中效果很好，因为所有 token 共享统一的语言语义空间。但在推荐场景中，特征空间天然异构——用户侧和物品侧的 ID 空间可能各自包含数亿个元素，计算这些异构语义空间之间的内积相似度本身就"没有明确意义"，容易引入噪声，还会带来更高的计算量、显存 IO 开销和显存占用。实验结果也验证了这一点（见下文消融实验，Self-Attention 路由的 AUC 反而略逊于 Token Mixing，同时 FLOPs 增加 71.8%）。

#### 2.3.2 Per-token FFN：参数隔离的特征子空间建模

传统 DLRM/DHEN 类模型往往把不同语义空间的特征塞进同一个交互模块共同处理，容易造成"高频特征域主导，长尾/低频特征信号被淹没"的问题。

RankMixer 提出 **Per-token FFN（PFFN）**：每个 Token 拥有**独立的、不共享的**一套 FFN 参数，而不是像标准 Transformer 那样所有 Token 共享同一个 FFN。对第 $t$ 个 Token $s_t$：

$$
v_t = f_{pffn}^{t,2}\Big(\mathrm{Gelu}\big(f_{pffn}^{t,1}(s_t)\big)\Big)
$$

其中第 $i$ 层线性变换为：

$$
f_{pffn}^{t,i}(x) = x W_{pffn}^{t,i} + b_{pffn}^{t,i}
$$

参数形状：

- $W_{pffn}^{t,1} \in \mathbb{R}^{D \times kD}$，$b_{pffn}^{t,1} \in \mathbb{R}^{kD}$
- $W_{pffn}^{t,2} \in \mathbb{R}^{kD \times D}$，$b_{pffn}^{t,2} \in \mathbb{R}^{D}$
- $k$ 为超参数，控制 FFN 隐藏维度相对 $D$ 的放大比例
- $\mathrm{Gelu}(\cdot)$ 为激活函数

整个 PFFN 模块可写作：

$$
v_1, \dots, v_T = \mathrm{PFFN}(s_1, \dots, s_T)
$$

**PFFN 与传统结构的本质区别**（论文特别强调）：

| 结构 | 输入 | 参数 |
|---|---|---|
| 标准 Transformer FFN | 不同 Token 各自输入 | 所有 Token **共享**一套 FFN 参数 |
| MMoE 中的 Expert | 所有 Expert **共享同一个输入** | 不同 Expert 参数不同 |
| RankMixer 的 Per-token FFN | 不同 Token **各自不同的输入** | 不同 Token **各自独立**的参数 |

也就是说，PFFN 是"**输入和参数同时按 Token 切分**"，这种设计天然契合"让不同特征子空间学到有差异化的表达"这一目标，在不增加计算复杂度（相对共享 FFN 而言，计算量不变，只是参数量增加）的前提下显著提升了模型容量。

#### 2.3.3 Sparse-MoE 变体：进一步提升 ROI

Per-token FFN 已经让参数量随 Token 数线性增长，如果想进一步扩容，一个自然的思路是把每个 Token 的 Dense FFN 换成 **Sparse Mixture-of-Experts（SMoE）**：模型容量继续增大，但计算量基本不变。

但论文发现，**朴素的 Sparse-MoE 在 RankMixer 里会"水土不服"**，原因有两点：

1. **均匀 Top-k 路由的问题**：Top-k 选择对所有 Token 一视同仁，把有限的专家预算浪费在低信息量 Token 上，同时"饿"到了高信息量 Token，抹杀了 Token 间信息密度的差异。
2. **专家训练不充分**：Per-token FFN 本身已经把参数按 Token 数做了倍增，如果再叠加不共享的 Expert，专家数量会进一步爆炸式增长，导致路由极度不均衡、专家训练不充分（"死亡专家"问题）。

为解决这两个问题，RankMixer 组合了两个训练策略：

**① ReLU Routing（替代 Top-k + Softmax）**

为了让不同 Token 能拥有灵活的、可变的专家激活数量，同时保持路由函数可微，RankMixer 用 **ReLU 门控 + 自适应 $\ell_1$ 惩罚**替代常见的 Top-k + Softmax 路由（该思路借鉴自 ReMoE）。

给定 Token $s_i \in \mathbb{R}^{d_h}$、第 $j$ 个专家 $e_{i,j}(\cdot)$、路由函数 $h(\cdot)$：

$$
G_{i,j} = \mathrm{ReLU}\big(h(s_i)\big), \qquad v_i = \sum_{j=1}^{N_e} G_{i,j}\, e_{i,j}(s_i)
$$

其中 $N_e$ 是每个 Token 可选的专家总数，$N_t$ 是 Token 总数。由于是 ReLU 而非 Top-k，门控值 $G_{i,j}$ 天然带有稀疏性（可以为 0），且对高信息量 Token 会自动激活更多专家。

稀疏度通过正则项 $L_{reg}$ 和系数 $\lambda$ 控制，使平均激活专家比例维持在预算附近：

$$
L = L_{task} + \lambda L_{reg}, \qquad L_{reg} = \sum_{i=1}^{N_t} \sum_{j=1}^{N_e} G_{i,j}
$$

**② DTSI-MoE：Dense-Training / Sparse-Inference**

借鉴文献 [Pan et al., 2024] 的思路，RankMixer 使用**两套路由器** $h_{train}$ 和 $h_{infer}$：

- 训练阶段，$h_{train}$ 和 $h_{infer}$ **都会**参与前向和梯度更新（保证所有专家都能获得充分梯度，即 Dense-Training）
- 稀疏正则 $L_{reg}$ **只作用于** $h_{infer}$
- 推理阶段，**只使用** $h_{infer}$ 做稀疏路由（Sparse-Inference）

这一组合的效果是：专家在训练阶段都能被充分训练，不会出现"死亡专家"，而在推理阶段又能享受稀疏激活带来的低成本。实验（论文 Figure 3）显示，即使把激活专家比例压缩到 1/8，DTSI + ReLU Routing 组合几乎不损失 1B 稠密模型的精度，同时推理吞吐提升 50%。

### 2.4 四个可扩展维度（Scaling 公式）

RankMixer 是一个高度并行、可扩展的架构，其参数量和计算量可以沿**四个正交的方向**扩展：

- Token 数量 $T$
- 模型宽度 $D$
- 层数 $L$
- 专家数量 $E$（Sparse-MoE 场景）

对于全稠密激活版本，单样本的参数量和前向 FLOPs 近似为：

$$
\#\mathrm{Param} \approx 2kLTD^2, \qquad \mathrm{FLOPs} \approx 4kLTD^2
$$

其中 $k$ 是调整 FFN 隐藏维度的缩放比例。而在 Sparse-MoE 版本中，每个 Token 的实际参数量和计算量会进一步被稀疏度 $s = \frac{\#\mathrm{Activated\_Param}}{\#\mathrm{Total\_Param}}$ 缩放。

论文实验发现一个和 LLM Scaling Law 一致的结论：**模型效果主要与总参数量相关，而深度 $L$、宽度 $D$、Token 数 $T$ 这几个不同的扩展方向带来的效果几乎相同**；但从计算效率角度看，增大隐藏维度 $D$ 能产生更大的矩阵乘形状，从而比单纯堆层数获得更高的 MFU。最终线上采用的配置为：

- **100M 版本**：$D=768,\ T=16,\ L=2$
- **1B 版本**：$D=1536,\ T=32,\ L=2$

### 2.5 消融实验揭示的架构价值

论文对 RankMixer-100M 做了详细消融（Table 2、Table 3），进一步印证了每个模块设计的必要性：

| 去掉的组件 | AUC 变化 |
|---|---|
| 残差连接（skip connections） | −0.07% |
| Multi-head Token Mixing | **−0.50%**（影响最大） |
| LayerNorm | −0.05% |
| Per-token FFN → 共享 FFN | −0.31% |

以及 Token→FFN 路由策略对比：

| 路由策略 | ΔAUC | ΔParams | ΔFLOPs |
|---|---|---|---|
| All-Concat-MLP（拼接后过大MLP再切分） | −0.18% | 0% | 0% |
| All-Share（不切分，全部输入共享，类似MMoE） | −0.25% | 0% | 0% |
| Self-Attention 路由 | −0.03% | +16% | **+71.8%** |
| **Multi-Head Token Mixing（RankMixer采用）** | — | — | — |

可以看到，去掉 Token Mixing 影响最大——因为一旦缺失，每个 FFN 只能看到局部特征、丧失全局信息交互能力；而 Self-Attention 虽然效果与 Token Mixing 接近，但代价是显著更高的参数量和计算量，验证了论文"异构特征空间上做内积相似度收益有限、代价高昂"的核心论点。


## 3 工程实现

### 3.1 特征 Tokenization 模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    将拼接后的异构特征 embedding 切分为 T 个固定维度的 token，
    并投影到统一的模型宽度 D。
    对应论文公式 (2): x_i = Proj(e_input[d*(i-1):d*i])
    """

    def __init__(self, input_dim: int, num_tokens: int, model_dim: int):
        super().__init__()
        assert input_dim % num_tokens == 0, "特征分组需能被 T 整除，否则请先padding"
        self.num_tokens = num_tokens
        self.chunk_dim = input_dim // num_tokens  # 对应论文中的 d
        self.model_dim = model_dim

        # 每个 token 有独立的投影矩阵（Proj 函数），
        # 也可以选择所有 token 共享一个 Proj，这里采用逐 token 独立投影
        self.proj = nn.ModuleList([
            nn.Linear(self.chunk_dim, model_dim) for _ in range(num_tokens)
        ])

    def forward(self, e_input: torch.Tensor) -> torch.Tensor:
        """
        e_input: [B, input_dim]  已经按语义分组拼接好的特征向量
        return:  [B, T, D]
        """
        chunks = e_input.split(self.chunk_dim, dim=-1)  # T 个 [B, chunk_dim]
        tokens = [proj(chunk) for proj, chunk in zip(self.proj, chunks)]
        return torch.stack(tokens, dim=1)  # [B, T, D]
```

### 3.2 Multi-Head Token Mixing（无参数特征交互）

这是全文实现中最关键、也最容易写错的部分。核心操作是：把每个 token 切成 H 份 head，然后**跨 token 拼接同一 head 位置**，形成新的 mixed token。用 `reshape + transpose` 即可高效实现，纯张量操作，不含任何可学习参数。

```python
class MultiHeadTokenMixing(nn.Module):
    """
    对应论文 3.3.1 节 & 公式 (3)(4)(5)。
    输入 X: [B, T, D] -> 输出 S: [B, T, D] （因为设定 H = T，token数不变）

    实现思路：
      1. 每个 token 切分为 H 个 head，每个 head 维度 D // H
      2. 按 head 维度重组：把所有 token 的第 h 个 head 拼在一起，形成新 token s^h
      3. 因为设定 H = T，输出 token 数与输入一致，天然可以做残差连接
    """

    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        self.T = num_tokens
        self.D = model_dim
        self.H = num_tokens  # 论文设定 H = T
        assert model_dim % self.H == 0, "D 必须能被 H 整除"
        self.head_dim = model_dim // self.H  # D // H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]  (H个mixed token，每个维度 T * (D//H) = D，
                             因为这里 H=T 所以 mixed token 维度恰好还是 D)
        """
        B, T, D = x.shape
        assert T == self.T and D == self.D

        # Step1: SplitHead -> [B, T, H, head_dim]
        x = x.view(B, T, self.H, self.head_dim)

        # Step2: 交换 T 和 H 维度，实现"跨 token 按 head 重组"
        # 变换前 x[b, t, h, :] 表示第 t 个token的第 h 个head
        # 变换后 s[b, h, t, :] 把同一个 h 下所有 t 的 head 排在一起
        s = x.transpose(1, 2).contiguous()  # [B, H, T, head_dim]

        # Step3: 把 (T, head_dim) merge 成一个维度，得到每个 mixed token 的向量
        # s^h = Concat(x_1^h, x_2^h, ..., x_T^h)  ->  维度 T * head_dim
        s = s.view(B, self.H, T * self.head_dim)  # [B, H, T*head_dim]

        # 因为 H = T 且 T*head_dim = T*(D//T)，一般设计中会让 T*(D//T) = D
        # 若 D 恰好能被 T 整除，则 mixed token 维度与输入 D 完全一致
        return s  # [B, T(=H), D]
```

> **实现小贴士**：论文要求 $H=T$ 是为了让 Token Mixing 前后 Token 数量保持一致，从而可以直接做残差连接（`+ X_{n-1}`）。在实现里，只要保证 `D % T == 0`，上面的 `view+transpose+view` 三步就能保证输出形状 `[B, T, D]` 与输入完全对齐，可以直接残差相加。这个操作本质上等价于对 `[B, T, H, head_dim]` 张量做一次 `transpose(1,2)`，是纯 tensor 重排，没有任何矩阵乘法，正是论文强调的"parameter-free"、GPU 上极其高效（几乎零计算开销，只有内存重排）。

### 3.3 Per-token FFN（Dense 版本）

```python
class PerTokenFFN(nn.Module):
    """
    对应论文 3.3.2 节 公式 (6)(7)(8)(9)。
    每个 token 拥有独立、不共享的两层 MLP 参数。
    用 batched matmul（一次性对所有 token 做不同的线性变换）实现，
    以保持GPU上的大GEMM并行效率。
    """

    def __init__(self, num_tokens: int, model_dim: int, k: int = 4):
        super().__init__()
        self.T = num_tokens
        self.D = model_dim
        self.hidden = k * model_dim

        # 用 [T, D, kD] 的参数张量表示"每个 token 独立的第一层权重"
        self.w1 = nn.Parameter(torch.empty(num_tokens, model_dim, self.hidden))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, self.hidden))
        self.w2 = nn.Parameter(torch.empty(num_tokens, self.hidden, model_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, model_dim))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [B, T, D]
        return v: [B, T, D]
        """
        # 第一层：对每个 token t，做 s_t @ W1_t + b1_t
        # 用 einsum 一次性对所有 token 做 batched matmul，映射到工程上就是
        # 融合成一个大 kernel（"fusing parallel per-token FFNs into one kernel"），
        # 这正是论文提升 MFU 的关键工程手段之一。
        h = torch.einsum('btd,tdh->bth', s, self.w1) + self.b1  # [B, T, hidden]
        h = F.gelu(h)
        v = torch.einsum('bth,thd->btd', h, self.w2) + self.b2  # [B, T, D]
        return v
```

### 3.4 Sparse-MoE 版 Per-token FFN：ReLU Routing + DTSI

```python
class Expert(nn.Module):
    """单个专家：标准两层MLP"""

    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class PerTokenSparseMoEFFN(nn.Module):
    """
    对应论文 3.4 节：ReLU Routing + DTSI-MoE (Dense-Training / Sparse-Inference)。

    关键点：
      1. 每个 token 有自己独立的一组专家（Per-token + MoE 组合）
      2. 路由用 ReLU(h(s_i)) 而不是 Top-k + Softmax，天然稀疏且可微
      3. 训练时使用两套路由器 h_train / h_infer，
         L_reg（稀疏正则）只作用于 h_infer，
         但两套路由器在训练阶段都会更新（Dense-Training）；
         推理阶段只使用 h_infer 做真正的稀疏计算（Sparse-Inference）
    """

    def __init__(self, num_tokens: int, model_dim: int, num_experts: int,
                 expert_hidden: int, reg_coef: float = 1e-3):
        super().__init__()
        self.T = num_tokens
        self.D = model_dim
        self.Ne = num_experts
        self.reg_coef = reg_coef

        # 每个 token 拥有自己独立的一组专家
        self.experts = nn.ModuleList([
            nn.ModuleList([Expert(model_dim, expert_hidden) for _ in range(num_experts)])
            for _ in range(num_tokens)
        ])

        # 两套路由器：训练用 / 推理用，每个 token 独立
        self.router_train = nn.Parameter(torch.empty(num_tokens, model_dim, num_experts))
        self.router_infer = nn.Parameter(torch.empty(num_tokens, model_dim, num_experts))
        nn.init.xavier_uniform_(self.router_train)
        nn.init.xavier_uniform_(self.router_infer)

        self.last_reg_loss = 0.0  # 供外部读取加到 total loss 里

    def _route(self, s: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        """
        s: [B, T, D], router: [T, D, Ne]
        return gate: [B, T, Ne]  (ReLU 之后, 天然带0，实现稀疏)
        对应公式(10): G_{i,j} = ReLU(h(s_i))
        """
        logits = torch.einsum('btd,tde->bte', s, router)
        return F.relu(logits)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, T, D = s.shape

        if self.training:
            # ---- Dense-Training：两套路由器都参与前向、都计算梯度 ----
            gate_train = self._route(s, self.router_train)   # [B,T,Ne]
            gate_infer = self._route(s, self.router_infer)   # [B,T,Ne]

            # 稀疏正则只施加在 infer 路由器上，对应公式(11)
            self.last_reg_loss = self.reg_coef * gate_infer.sum(dim=(1, 2)).mean()

            out_train = self._compute_experts_output(s, gate_train)
            out_infer = self._compute_experts_output(s, gate_infer)

            # 两路输出都参与训练（简单相加/平均均可，具体融合方式工程上可调），
            # 保证所有专家都获得充分梯度，避免"死亡专家"
            return 0.5 * (out_train + out_infer)
        else:
            # ---- Sparse-Inference：只用 infer 路由器，且可结合门控值做真稀疏跳过计算 ----
            gate_infer = self._route(s, self.router_infer)
            self.last_reg_loss = 0.0
            return self._compute_experts_output(s, gate_infer, sparse_exec=True)

    def _compute_experts_output(self, s: torch.Tensor, gate: torch.Tensor,
                                 sparse_exec: bool = False) -> torch.Tensor:
        """
        对应公式(10): v_i = sum_j G_{i,j} * e_{i,j}(s_i)
        sparse_exec=True 时，跳过 gate==0 的专家计算以节省真实推理算力
        （工程上通常按 batch 聚合非零 mask 做 gather/scatter，
         这里为了可读性用简单实现）
        """
        B, T, D = s.shape
        outputs = torch.zeros_like(s)
        for t in range(T):
            token_in = s[:, t, :]              # [B, D]
            token_gate = gate[:, t, :]          # [B, Ne]
            token_out = torch.zeros_like(token_in)
            for e_idx, expert in enumerate(self.experts[t]):
                g = token_gate[:, e_idx].unsqueeze(-1)  # [B, 1]
                if sparse_exec:
                    active_mask = (g.squeeze(-1) > 0)
                    if not active_mask.any():
                        continue  # 该专家在这个 batch 里对所有样本都未激活，跳过计算
                    active_idx = active_mask.nonzero(as_tuple=True)[0]
                    expert_out = expert(token_in[active_idx])
                    token_out[active_idx] += g[active_idx] * expert_out
                else:
                    token_out += g * expert(token_in)
            outputs[:, t, :] = token_out
        return outputs
```

> **说明**：真实工业实现中，为了拿到论文所述"融合并行 kernel、大 GEMM 形状"的高 MFU 收益，逐 token / 逐专家 for 循环会被替换为**分组批量矩阵乘（grouped GEMM）+ 稀疏 gather/scatter kernel**（例如基于 CUTLASS 或 Triton 自定义 kernel），并结合 fp16/bf16 量化推理。上面的实现优先保证与论文公式的一一对应关系、便于理解，工程落地时的优化方向见下一节。

### 3.5 完整的 RankMixer Block 与堆叠

```python
class RankMixerBlock(nn.Module):
    """
    对应论文公式(1):
      S_{n-1} = LN(TokenMixing(X_{n-1}) + X_{n-1})
      X_n     = LN(PFFN(S_{n-1}) + S_{n-1})
    """

    def __init__(self, num_tokens: int, model_dim: int, k: int = 4,
                 use_moe: bool = False, num_experts: int = 4, expert_hidden: int = None):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens, model_dim)
        self.ln1 = nn.LayerNorm(model_dim)

        if use_moe:
            expert_hidden = expert_hidden or k * model_dim
            self.ffn = PerTokenSparseMoEFFN(num_tokens, model_dim, num_experts, expert_hidden)
        else:
            self.ffn = PerTokenFFN(num_tokens, model_dim, k)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.ln1(self.token_mixing(x) + x)
        out = self.ln2(self.ffn(s) + s)
        return out


class RankMixer(nn.Module):
    """
    完整的 RankMixer 模型：Tokenization -> L层Block -> mean pooling -> 多任务头
    """

    def __init__(self, input_dim: int, num_tokens: int, model_dim: int,
                 num_layers: int, k: int = 4, use_moe: bool = False,
                 num_experts: int = 4, task_names=("finish", "skip", "like")):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, num_tokens, model_dim)
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, model_dim, k, use_moe, num_experts)
            for _ in range(num_layers)
        ])
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(model_dim, 1) for name in task_names
        })

    def forward(self, e_input: torch.Tensor) -> dict:
        x = self.tokenizer(e_input)          # [B, T, D]
        for block in self.blocks:
            x = block(x)                     # [B, T, D]
        pooled = x.mean(dim=1)               # mean pooling -> [B, D]
        return {name: torch.sigmoid(head(pooled)).squeeze(-1)
                for name, head in self.task_heads.items()}


# ------------------- 简单自测 -------------------
if __name__ == "__main__":
    B, T, D, L = 8, 16, 768, 2   # 对应论文100M配置 D=768,T=16,L=2
    input_dim = T * 64           # 假设每个token切分前维度d=64

    model = RankMixer(input_dim=input_dim, num_tokens=T, model_dim=D,
                       num_layers=L, k=4, use_moe=False)
    dummy_input = torch.randn(B, input_dim)
    outputs = model(dummy_input)
    for task, pred in outputs.items():
        print(task, pred.shape)  # 期望输出: torch.Size([8])
```

### 3.6 工程优化要点

论文给出了一个非常实用的时延分解公式：

$$
\mathrm{Latency} = \frac{\#\mathrm{Param} \times \mathrm{FLOPs/Param\ ratio}}{\mathrm{MFU} \times \mathrm{Theoretical\ Hardware\ FLOPs}}
$$

字节在把参数从 16M 扩到 1.1B（约 70 倍）的过程中，时延几乎持平（14.5ms → 14.3ms），依赖三方面工程手段：

1. **降低 FLOPs/Param 比值（3.6× 收益）**：架构设计本身（Per-token FFN 保持计算量不变、只增加参数）使得参数增长 70 倍，FLOPs 只增长约 20.7 倍。
2. **提升 MFU（约 10× 收益）**：
   - 使用大 GEMM 形状（更宽的 $D$）
   - 良好的并行拓扑：**把并行的多个 Per-token FFN 融合进同一个 kernel**（对应上文 `einsum` 批量实现思路的工业级落地——用 batched/grouped GEMM 一次性计算所有 token 的 FFN，而不是 Python for 循环）
   - 降低访存带宽开销与调度开销，让模型从 Memory-bound 转为 Compute-bound
3. **量化（2× 收益）**：RankMixer 的主要计算是若干个大矩阵乘法，非常适合 **fp16 半精度推理**，直接将理论峰值算力提升一倍。

## 4 总结

RankMixer 的核心贡献可以概括为一句话：**用"硬件对齐 + 数据特性对齐"这两条准则重新设计了推荐排序模型的特征交互架构**，从而真正打通了 Scaling Law 在工业推荐系统中的落地路径。

具体来看：

1. **架构层面**：用无参数的 **Multi-Head Token Mixing** 替代平方复杂度、且在异构特征空间上语义不明确的 Self-Attention，用极低成本完成全局特征交互；用 **Per-token FFN** 实现"输入与参数同步按特征子空间隔离"，在不增加计算量的前提下大幅提升模型容量，天然避免了高频特征淹没长尾特征的问题；进一步用 **ReLU Routing + DTSI（Dense-Training/Sparse-Inference）** 的 Sparse-MoE 变体，在几乎不损失精度的情况下把参数容量再放大数倍。
2. **Scaling 层面**：论文证明了 RankMixer 在 Token 数 $T$、模型宽度 $D$、层数 $L$、专家数 $E$ 四个维度上都具有良好且相近的扩展效果，效果主要取决于总参数量本身，这与 LLM 领域观察到的 Scaling Law 规律高度一致。
3. **工程层面**：通过降低 FLOPs/Param 比值、大幅提升 MFU（融合 kernel、大 GEMM 形状）、以及 fp16 量化，三管齐下抵消了参数量 70 倍增长带来的成本压力，做到了"参数涨、时延不涨"。
4. **业务价值**：RankMixer-1B 已经全量上线抖音 Feed 精排和电商广告精排等数十个字节内部场景，线上 A/B 测试证明其在推荐和广告两大核心场景中都具有良好的通用性，且低活用户群体的收益尤其显著（Active Days +1.74%），验证了大模型对稀疏样本用户的泛化增益。

从更宏观的视角看，RankMixer 代表了推荐系统排序模型的一种范式转变：从"堆砌人工设计的特征交叉算子"转向"设计硬件友好、可无限堆叠的统一架构"，这与 LLM 领域从"手工特征工程"走向"Transformer + Scaling Law"的历史进程颇为相似。可以预见，随着 Sparse-MoE 进一步成熟，RankMixer 有望从当前的 1B 参数规模继续扩展到未来 10B 级别的工业推荐场景。
