# 多门控混合专家模型（Multi-gate Mixture-of-Experts，MMoE）

MMoE（Multi-gate Mixture-of-Experts）是 Google 于 2018 年提出的多任务学习模型，发表于 KDD 2018，论文题为 *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*。[1]

MMoE 的核心思想是通过**共享专家网络（Expert Networks）与任务独立的门控网络（Gate Networks）**，动态学习不同任务之间的共享关系，在实现知识共享的同时保留任务差异性。

## 1. 背景（Background）

多任务学习（Multi-task Learning，MTL）通过共享模型参数，同时优化多个相关任务，例如推荐系统中的点击率（CTR）和转化率（CVR）预测。传统 Shared-Bottom 结构采用共享底层网络和任务独立预测层，但当任务相关性较低时，容易产生负迁移（Negative Transfer），影响模型性能。

MMoE 针对这一问题，引入多个共享专家网络，并为每个任务设计独立的门控网络。不同任务可以根据输入动态选择专家的组合权重，从而实现更灵活的参数共享，缓解任务间的负迁移问题。

## 2. 模型架构（Model Architecture）

### 2.1 整体网络结构

![MMoE架构图](../img/mmoe-architecture.svg)

MMoE 由三个核心部分组成：

- **Expert Network**：多个共享专家网络，用于学习不同的特征表示。
- **Gate Network**：每个任务拥有独立的门控网络，控制各专家的贡献权重。
- **Task Tower**：每个任务拥有独立的预测网络，输出对应任务的预测结果。

输入特征首先送入多个共享 Expert，每个 Expert 学习一种潜在的特征变换方式；随后，不同任务通过各自的 Gate 对这些 Expert 的输出进行加权组合，从而形成任务特定的表示；最后，各任务将对应表示输入各自的 Tower，输出最终预测结果。

这种结构的核心在于：共享 Expert 负责知识共享，独立 Gate 负责任务选择，独立 Tower 负责任务输出。因此，MMoE 能够在共享表示学习的基础上，更灵活地建模任务之间的相关性与差异性。

### 2.2 从 Shared-Bottom 到 MMoE

为了理解 MMoE 的设计动机，可以先比较三种多任务学习架构。

**（1）Shared-Bottom**

Shared-Bottom 是典型的硬参数共享（Hard Parameter Sharing）结构，多个任务共享一个底层特征提取网络：

$$
h=f(x)
$$

$$
y_k=t_k(h)
$$

其中：

- $x$：模型输入特征。
- $f(\cdot)$：共享底层网络。
- $h$：共享特征表示。
- $t_k(\cdot)$：第 $k$ 个任务的专属网络。

所有任务使用相同的底层表示，仅在任务预测层进行区分。

这种方式结构简单、参数共享程度高，但不同任务的梯度会共同更新共享网络。当任务优化方向存在冲突时，可能出现负迁移。

**（2）One-gate Mixture-of-Experts（OMoE）**

OMoE 使用多个专家网络代替单一共享网络，并通过一个公共 Gate 动态融合专家输出。

设共有 $n$ 个专家：

$$
E(x)=[e_1(x),e_2(x),\ldots,e_n(x)]
$$

其中 $e_i(x)$ 表示第 $i$ 个专家网络的输出。

公共 Gate 计算专家权重：

$$
g(x)=\operatorname{softmax}(W_gx)
$$

融合后的特征表示为：

$$
h(x)=\sum_{i=1}^{n}g_i(x)e_i(x)
$$

其中：

$$
\sum_{i=1}^{n}g_i(x)=1
$$

OMoE 能够根据输入样本动态调整专家组合，但所有任务仍使用同一个 Gate，因而对于同一个输入，不同任务得到相同的专家融合表示。

**（3）Multi-gate Mixture-of-Experts（MMoE）**

MMoE 在 OMoE 的基础上，为每个任务配置独立的 Gate。

对于第 $k$ 个任务：

$$
g^k(x)=\operatorname{softmax}(W_{gk}x)
$$

对应的任务特征表示为：

$$
h^k(x)=\sum_{i=1}^{n}g_i^k(x)e_i(x)
$$

最终预测结果为：

$$
y_k=t_k(h^k(x))
$$

与 OMoE 不同，MMoE 中不同任务具有独立的专家组合权重：

$$
g^1(x)\neq g^2(x)
$$

这里表示两者不要求相等，而非对所有输入都必然不同。

因此，即使两个任务共享同一组 Expert，也可以得到不同的任务特征表示。

三种结构的核心区别如下：

| 模型 | Expert | Gate | 任务特征表示 |
|---|---|---|---|
| Shared-Bottom | 单个共享网络 | 无 | 全任务共享 |
| OMoE | 多个共享专家 | 全任务共享一个 | 全任务共享 |
| MMoE | 多个共享专家 | 每个任务独立 | 各任务动态生成 |

MMoE 的主要创新并不是简单增加专家数量，而是通过**任务独立的 Gate 实现任务相关性的自适应建模**。

### 2.3 专家网络（Expert Network）

Expert Network 是 MMoE 的共享特征提取模块。

假设模型包含 $n$ 个 Expert，每个 Expert 都是一个独立的神经网络：

$$
e_i(x)=f_i(x;\theta_i)
$$

其中：

- $x\in\mathbb{R}^{d}$：输入特征向量。
- $f_i$：第 $i$ 个专家网络。
- $\theta_i$：对应专家网络的参数。
- $e_i(x)\in\mathbb{R}^{h}$：专家输出向量。

在实际实现中，Expert 通常使用多层感知机（MLP），例如：

$$
e_i(x)=\operatorname{ReLU}(W_{i,2}\operatorname{ReLU}(W_{i,1}x+b_{i,1})+b_{i,2})
$$

所有专家接收相同的输入，但参数互不相同。

通过联合训练，各专家可以学习不同的特征变换方式，为不同任务提供多样化的表示。

需要注意，MMoE 并没有强制规定各个 Expert 必须学习特定语义。专家之间的分工是训练过程中形成的，并不保证每个 Expert 都具有明确、独立的功能。

### 2.4 门控网络（Gate Network）

Gate Network 是 MMoE 最关键的模块。

传统 MoE 使用一个公共 Gate，而 MMoE 为每个任务设置独立 Gate，使任务能够根据自身需求动态选择 Expert。

对于第 $k$ 个任务，Gate 的计算公式为：

$$
g^k(x)=\operatorname{softmax}(W_{gk}x)
$$

其中：

$$
W_{gk}\in\mathbb{R}^{n\times d}
$$

Gate 输出为：

$$
g^k(x)\in\mathbb{R}^{n}
$$

其第 $i$ 个分量表示第 $k$ 个任务对第 $i$ 个 Expert 分配的权重：

$$
g_i^k(x)=
\frac{\exp((W_{gk}x)_i)}
{\sum_{j=1}^{n}\exp((W_{gk}x)_j)}
$$

Softmax 保证每个权重非负，并且所有专家权重之和为 1。

假设模型包含三个 Expert，分别输出：

$$
e_1(x)=[0.2,0.5]
$$

$$
e_2(x)=[0.8,0.3]
$$

$$
e_3(x)=[0.4,0.9]
$$

对于 CTR 任务，Gate 输出：

$$
g^{CTR}(x)=[0.6,0.3,0.1]
$$

对于 CVR 任务，Gate 输出：

$$
g^{CVR}(x)=[0.1,0.2,0.7]
$$

则 CTR 的融合特征为：

$$
\begin{aligned}
h^{CTR}(x)
&=0.6e_1(x)+0.3e_2(x)+0.1e_3(x)\\
&=[0.40,0.48]
\end{aligned}
$$

CVR 的融合特征为：

$$
\begin{aligned}
h^{CVR}(x)
&=0.1e_1(x)+0.2e_2(x)+0.7e_3(x)\\
&=[0.46,0.74]
\end{aligned}
$$

可以看到，虽然两个任务使用同一组专家，但由于 Gate 权重不同，最终得到的特征表示也不同。

此外，Gate 权重由输入 $x$ 动态生成，因此同一个任务面对不同样本时，也可以采用不同的专家组合。

MMoE 同时实现了两个层面的自适应：

- **任务级自适应**：不同任务使用不同的 Gate 参数。
- **样本级自适应**：同一任务根据不同输入生成不同的专家权重。

### 2.5 任务塔（Task Tower）

经过 Gate 加权融合后，每个任务得到自己的特征表示：

$$
h^k(x)=\sum_{i=1}^{n}g_i^k(x)e_i(x)
$$

随后将其输入任务独立的 Tower：

$$
\hat y_k=t_k(h^k(x))
$$

Tower 通常由一层或多层 MLP 构成，负责学习任务特定的特征映射。

对于 CTR 和 CVR 等二分类任务，可以使用 Sigmoid 输出概率：

$$
\hat y_k=\sigma(f_k(h^k(x)))
$$

其中：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

需要注意，MMoE 仅规定任务通过独立的 Tower 完成预测，并不限制所有任务必须采用相同的输出形式。

例如，分类任务可以使用 Sigmoid，回归任务则可以直接输出连续数值。

### 2.6 损失函数（Loss Function）

MMoE 本身是一种多任务特征共享架构，并没有引入必须使用的特殊损失函数。

通常使用多个任务损失的加权和进行联合训练：

$$
\mathcal{L}=
\sum_{k=1}^{K}\lambda_k\mathcal{L}_k
$$

其中：

- $K$：任务数量。
- $\mathcal{L}_k$：第 $k$ 个任务的损失函数。
- $\lambda_k$：对应任务的损失权重。

对于 CTR 和 CVR 两个二分类任务，可以分别使用二元交叉熵（Binary Cross-Entropy，BCE）：

$$
\mathcal{L}_{CTR}
=
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i^{CTR}\log\hat y_i^{CTR}
+
(1-y_i^{CTR})\log(1-\hat y_i^{CTR})
\right]
$$

CVR 任务同样可以采用 BCE，其最终损失为：

$$
\mathcal{L}
=
\lambda_{CTR}\mathcal{L}_{CTR}
+
\lambda_{CVR}\mathcal{L}_{CVR}
$$

联合训练时，Expert 接收多个任务损失传播的梯度，而每个 Gate 和 Tower 主要由对应任务的损失更新。

这使得 Expert 学习跨任务共享表示，Gate 和 Tower 则保留任务特定的建模能力。

需要注意：如果 CVR 被定义为点击后转化概率，实际数据中通常需要对未点击样本进行标签掩码或采用其他合适的训练目标，不能直接将全部未点击样本当作未转化样本。本节的公式仅表示一般的多任务二分类训练形式。

## 3. 模型实现（Implementation）

### 3.1 核心代码实现

下面使用 PyTorch 实现一个通用的 MMoE 模型，包括：

- 多个共享 Expert。
- 每个任务独立的 Gate。
- 每个任务独立的 Tower。
- 支持多个二分类任务的预测。

**（1）MMoE 模型定义**

```python
import torch
import torch.nn as nn


class MMoE(nn.Module):
    def __init__(
        self,
        input_dim,
        expert_dim=32,
        num_experts=4,
        num_tasks=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # 共享专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU()
            )
            for _ in range(num_experts)
        ])

        # 每个任务独立的 Gate
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # 每个任务独立的 Tower
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # [B, E, D]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )

        outputs = []

        for gate, tower in zip(
            self.gates, self.towers
        ):
            # [B, E]
            gate_weights = torch.softmax(
                gate(x), dim=-1
            )

            # [B, 1, E] @ [B, E, D]
            # -> [B, D]
            task_input = torch.bmm(
                gate_weights.unsqueeze(1),
                expert_outputs
            ).squeeze(1)

            # [B]
            logits = tower(task_input).squeeze(-1)
            outputs.append(logits)

        return outputs
```

其中：

- $B$：Batch Size。
- $E$：Expert 数量。
- $D$：Expert 输出维度。

模型的核心计算为：

```python
task_input = torch.bmm(
    gate_weights.unsqueeze(1),
    expert_outputs
).squeeze(1)
```

这一操作与数学公式完全对应：

$$
h^k(x)=\sum_{i=1}^{n}g_i^k(x)e_i(x)
$$

通过批量矩阵乘法，可以对当前任务的所有 Expert 输出进行加权求和。

**（2）模型前向计算**

下面使用两个任务模拟 CTR 和 CVR 预测：

```python
model = MMoE(
    input_dim=64,
    expert_dim=32,
    num_experts=4,
    num_tasks=2
)

x = torch.randn(8, 64)

ctr_logits, cvr_logits = model(x)

ctr_pred = torch.sigmoid(ctr_logits)
cvr_pred = torch.sigmoid(cvr_logits)

print("CTR:", ctr_pred.shape)
print("CVR:", cvr_pred.shape)
```

输出张量形状：

```text
CTR: torch.Size([8])
CVR: torch.Size([8])
```

这里的输入是假设已经完成特征编码和拼接的 64 维稠密特征，实际推荐系统可以在 MMoE 前增加类别特征 Embedding 等输入处理模块。

**（3）模型训练**

MMoE 可以直接使用多个任务损失的加权和进行反向传播：

```python
import torch.nn.functional as F

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 演示用模拟标签，不代表真实 CTR/CVR 数据关系
ctr_labels = torch.randint(
    0, 2, (8,)
).float()

cvr_labels = torch.randint(
    0, 2, (8,)
).float()

ctr_logits, cvr_logits = model(x)

ctr_loss = F.binary_cross_entropy_with_logits(
    ctr_logits, ctr_labels
)

cvr_loss = F.binary_cross_entropy_with_logits(
    cvr_logits, cvr_labels
)

loss = 0.5 * ctr_loss + 0.5 * cvr_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这里的两个任务都使用完整样本上的二分类损失，仅用于演示多任务联合训练。实际实现需要根据各任务的标签定义、样本有效性和业务目标确定损失计算方式。

## 4. 总结（Conclusion）

MMoE 通过共享 Expert 和任务独立 Gate，将传统多任务学习中的固定参数共享改进为动态的专家组合。相比 Shared-Bottom 和 OMoE，MMoE 能够为不同任务生成不同的特征表示，从而更灵活地建模任务之间的共享关系。

MMoE 的主要优势在于能够根据输入动态分配专家权重，在任务相关性较低时有助于缓解负迁移。原论文通过合成数据、公开分类数据及 Google 的大规模内容推荐任务验证了模型的有效性。[1]

不过，MMoE 并不能完全消除任务冲突。当任务数量或 Expert 数量增加时，模型的计算开销也会相应增加。此外，任务损失权重、专家数量以及不同 Gate 的学习效果都会影响最终性能。MMoE 因而适用于需要联合优化多个目标的推荐场景，也是理解后续多任务专家网络结构的重要基础。

## 5. 参考文献（References）

[1] Ma J, Zhao Z, Yi X, et al. **Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts**. KDD, 2018.  
https://doi.org/10.1145/3219819.3220007

[2] Jacobs R A, Jordan M I, Nowlan S J, Hinton G E. **Adaptive Mixtures of Local Experts**. Neural Computation, 1991.  
https://doi.org/10.1162/neco.1991.3.1.79

[3] Caruana R. **Multitask Learning**. Machine Learning, 1997.  
https://doi.org/10.1023/A:1007379606734

[4] Google Research. **Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts**.  
https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/

[5] PyTorch. **PyTorch Documentation**.  
https://docs.pytorch.org/docs/stable/index.html
