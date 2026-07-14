# 深入理解深度兴趣网络Deep Interest Network (DIN)

## 1 背景

在工业界推荐系统中，用户特征（User Profile）和历史行为（User Behavior History）是预测其未来点击倾向的核心依据。传统的深度预估网络通常遵循如下范式：

1. **Embedding 层**：将高维稀疏的类别特征（如商品 ID、店铺 ID）映射到低维稠密向量空间。
2. **Pooling 层**：对于变长的历史行为序列，通过 Sum-Pooling 或 Average-Pooling 转化为固定长度的特征向量。
3. **MLP 层**：拼接用户特征、候选广告特征（Candidate Ad）及上下文特征，送入多层感知机进行非线性交叉，最终输出点击概率。

$$\mathbf{x}_u = \text{Pooling}(\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n)$$

然而，这种设计存在两大核心缺陷：

- **兴趣压缩瓶颈（Limited Representation Capability）**：用户的兴趣是广泛且多样的（例如一个用户可能同时喜欢电子产品、运动鞋和书籍）。用一个固定长度的向量 $\mathbf{x}_u$ 去强行压缩用户所有的历史兴趣，随着历史序列的拉长，这会成为严重的信息表达瓶颈。
- **缺乏局部相关性捕捉（Lack of Local Activation）**：在面对具体的候选广告（Candidate Ad）时，用户表现出的兴趣往往是局部的。例如，当候选广告是一双“篮球鞋”时，用户历史行为中购买“键盘”和“小说”的记录应该被赋予极低的关注，而购买“运动外套”的记录应该被激活。传统的 Pooling 机制无视了候选广告的区别，对所有历史行为一视同仁，引入了大量噪声。

基于上述痛点，DIN 的核心动机非常直观：**不应该用一个固定的向量来表征用户所有的兴趣，而应该针对不同的候选广告，自适应地计算出用户与当前广告相关的“局部兴趣表达”。**


## 2 模型架构 (Model Architecture)

DIN 并没有颠覆传统的 Embedding & MLP 骨架，而是优雅地在用户历史行为序列与候选广告之间引入了一个**兴趣激活模块（Activation Unit）**。

### 2.1 基础网络拓扑对比

对比传统 Base 模型，DIN 在特征表示层进行了本质的革新。其整体前向计算流程如下：

- **Input & Embedding**：
将输入特征划分为四个核心组：User Profile Features, User Behavior Features, Candidate Ad Features, Context Features。每个特征被映射为低维 Embeddings。
- **Local Activation Unit（核心创新）**：
输入包括两部分：用户历史行为序列的 Embedding $\mathbf{e}_i$ 和候选广告的 Embedding $\mathbf{v}_A$。该模块为每个历史行为计算一个标量权重 $a(\mathbf{e}_i, \mathbf{v}_A)$，代表该行为与当前广告的相关性。

### 2.2 兴趣激活模块 (Activation Unit) 

不同于自然语言处理（NLP）中标准的注意力机制（Softmax Attention），DIN 的 Activation Unit 放宽了权重的约束。其计算公式如下：

$$\mathbf{v}_U(A) = f(\mathbf{v}_A, \mathcal{H}) = \sum_{i=1}^{T} a(\mathbf{e}_i, \mathbf{v}_A) \mathbf{e}_i = \sum_{i=1}^{T} w_i \mathbf{e}_i$$

其中，$\mathcal{H} = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_T\}$ 是长度为 $T$ 的用户历史行为向量集合。权重 $w_i = a(\mathbf{e}_i, \mathbf{v}_A)$ 是通过一个小型的全连接前馈网络输出的，该网络的输入除了 $\mathbf{e}_i$ 和 $\mathbf{v}_A$ 之外，还包含了它们的**外积（Out Product / Element-wise Subtraction & Multiplication）**，用以显式建模特征显式交叉。

> **⚠️ 注意：为什么不用标准的 Softmax？**
> 论文特别指出，DIN 没有对权重 $w_i$ 进行 Softmax 归一化。因为用户历史行为的绝对数量和强度本身就蕴含了重要信息。例如，一个在历史里点击过 10 次运动鞋的用户，其兴趣强度理应大于只点击过 1 次的用户。Softmax 会抹去这种序列长度带来的绝对强度差异。

### 2.3 自适应激活函数：Dice

传统的 ReLU 在输入小于 0 时导数为 0，容易导致神经元“坏死”；PReLU 虽有改善，但其修正点硬性固定在 0 处。针对推荐系统中工业级数据的多模态分布，DIN 提出了 **Dice (Data-Dependent Activation Unit)** 激活函数，其形式为：

$$f(s) = p(s) \cdot s + (1 - p(s)) \cdot \alpha s$$

$$p(s) = \frac{1}{1 + \exp\left(-\frac{s - E[s]}{\sqrt{Var[s] + \epsilon}}\right)}$$

Dice 核心思想是**根据输入数据的均值 $E[s]$ 和方差 $Var[s]$ 动态调整修正点的位置**，使得控制函数 $p(s)$ 随数据分布平滑移动，极大增强了大规模稀疏特征下的泛化能力。

## 3 工程实现与优化 (Engineering Implementation)

在工业级生产环境中，海量稀疏特征（如阿里巴巴场景下动辄百亿级参数）直接投入深度网络训练，极易引发过拟合以及单机计算瓶颈。DIN 提出了两项重大的工程优化手段。

### 3.1 自适应正则化 (Mini-batch Aware Regularization)

传统的 $L_2$ 正则化需要计算全量参数，但在稀疏场景下，每个 Mini-batch 仅有极少量的特征被激活。如果对未出现的特征也强行进行全量反向传播更新 $L_2$ 惩罚，不仅计算量爆炸，还会破坏原本未激活特征的 embedding 状态。

DIN 创新地提出了 **Mini-batch Aware Regularization**。每次迭代中，只对当前 Batch 中**实际出现过**的特征参数计算正则化惩罚，并将惩罚项根据该特征在全局数据中的频次进行加权缩放：

$$L_2(\mathbf{W}) \approx \sum_{m=1}^{M} \sum_{(x,y) \in \mathcal{B}_m} \sum_{j \in \mathcal{P}_x} \frac{I(x_j \neq 0)}{S_j} \Vert{}\mathbf{w}_j\Vert{}_2^2$$

其中 $S_j$ 为特征 $j$ 在全局样本中的总出现频次，$\mathcal{B}_m$ 为当前第 $m$ 个 mini-batch。这使得高频特征和低频特征在正则化中得到了公平的约束，大幅减少了训练波动。

### 3.2 核心伪代码实现 (PyTorch 风格)

以下是使用 PyTorch 实现 DIN 核心逻辑（Activation Unit）的简化工业参考代码：

```python
import torch
import torch.nn as nn

class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionSequencePoolingLayer, self).__init__()
        # 激活单元内部的 MLP 结构
        self.local_activation = nn.Sequential(
            nn.Linear(4 * embedding_dim, 80),
            nn.Dice(80), # 论文提出的自适应激活函数
            nn.Linear(80, 40),
            nn.Dice(40),
            nn.Linear(40, 1)
        )

    def forward(self, query, keys, keys_length):
        """
        query: [Batch_size, 1, embedding_dim] - 候选 Ad 的 embedding
        keys:  [Batch_size, Seq_len, embedding_dim] - 用户历史行为序列的 embeddings
        keys_length: [Batch_size] - 每个用户序列的实际真实长度
        """
        batch_size, max_seq_len, embedding_dim = keys.size()
        
        # 将 query 复制扩展到与 keys 相同的序列长度
        query = query.expand(-1, max_seq_len, -1) # [Batch_size, Seq_len, embedding_dim]
        
        # 构造混合交叉特征: query, key, query-key, query*key
        din_all = torch.cat([query, keys, query - keys, query * keys], dim=-1) # [Batch_size, Seq_len, 4 * embedding_dim]
        
        # 计算注意力得分
        outputs = self.local_activation(din_all) # [Batch_size, Seq_len, 1]
        outputs = outputs.squeeze(-1) # [Batch_size, Seq_len]
        
        # 制作 Padding Mask 消除填充零的影响
        mask = torch.arange(max_seq_len, device=keys.device)[None, :] < keys_length[:, None]
        outputs = outputs.masked_fill(~mask, -1e9) # 未达真实长度的部分用极小值填充
        
        # 转换成权重权重（注意：这里是否用 softmax 取决于业务选择，DIN 原文推荐直接用 exp/或不加限制的激活值，此处以原生乘法为例）
        a = torch.sigmoid(outputs).unsqueeze(1) # [Batch_size, 1, Seq_len]
        
        # 加权求和 (Weighted Sum Pooling)
        user_interest_vector = torch.bmm(a, keys) # [Batch_size, 1, embedding_dim]
        return user_interest_vector.squeeze(1)

```

## 4 总结

DIN 的主要贡献

- **局部激活的引入**：打破了传统深度学习将用户兴趣死板地压缩为单一向量的藩篱，首次将 Attention 机制成功落地于推荐系统的长尾序列建模中。
- **工程创新的表率**：不仅停留在理论公式，Dice 激活函数与 Mini-batch 正则化完美解决了大规模稀疏推荐场景下的过拟合与性能瓶颈。
