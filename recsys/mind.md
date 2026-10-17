# MIND：动态路由多兴趣网络

MIND（Multi-Interest Network with Dynamic Routing）发表于 CIKM 2019，论文题为 *Multi-Interest Network with Dynamic Routing for Recommendation at Tmall* [1]。

MIND 面向推荐系统的召回阶段，用多个向量分别表达用户的不同兴趣。模型通过行为到兴趣动态路由聚合历史行为，训练时使用标签感知注意力选择与目标物品相关的兴趣，服务时则让多个兴趣向量分别检索候选物品。

## 1. 背景

召回阶段需要从大规模物品库中快速找出少量候选。常见双塔模型将用户历史压缩为单个向量 $v_u$，再与物品向量 $e_i$ 计算相似度：

$$
s(u,i)=v_u^Te_i
$$

单向量表示容易混合用户的多种偏好。例如，一个用户同时喜欢跑步、烘焙和科幻小说，三个方向在嵌入空间中未必能被一个向量准确覆盖。折中的用户向量可能离每个兴趣对应的物品都不够近，从而降低召回覆盖率。

[DIN](/recsys/din/)、[DIEN](/recsys/dien/) 和 [DSIN](/recsys/dsin/) 都根据候选物品生成目标相关兴趣，适合已有候选集合后的精排。MIND 需要面对完整物品库，因此不能为每个物品重新计算一遍用户网络。它先生成少量候选无关的兴趣向量，再使用向量索引并行检索。

## 2. 模型架构

### 2.1 整体网络结构

![MIND架构图](/img/mind-architecture.svg)

> MIND架构图

MIND 包含四个主要部分：

1. **嵌入与池化层**：编码用户属性、历史物品和训练目标物品。
2. **多兴趣提取层**：通过行为到兴趣动态路由，将历史行为聚合为多个兴趣胶囊。
3. **用户表示网络**：将用户属性与每个兴趣向量结合并投影到物品向量空间。
4. **标签感知注意力**：训练时根据目标物品选择或组合兴趣向量，并使用采样 Softmax 优化。

服务时移除标签感知注意力。每个用户兴趣向量分别进行近邻检索，最后合并、去重并截取候选集合。

### 2.2 输入与嵌入

设用户历史包含 $L$ 个有效物品，行为嵌入为：

$$
E_u=[e_1,e_2,\ldots,e_L],\qquad e_t\in\mathbb R^D
$$

一个物品可以由物品 ID、类别 ID、品牌 ID 等字段组成。各字段查表后进行平均池化或其他固定聚合，得到统一的 $D$ 维物品表示。用户性别、年龄等属性嵌入拼接为用户画像向量 $p_u$。

历史物品和目标物品必须使用相同的物品编码网络，服务端向量索引也必须由同一版本的物品塔生成。补零 ID 不参与字段池化、动态路由或训练损失。

### 2.3 胶囊与压缩函数

MIND 把每条历史行为看作低层胶囊，把用户兴趣看作高层胶囊。与标量神经元不同，胶囊由向量表示；不同方向编码兴趣属性，向量长度被限制在小于 1 的范围。

给定聚合向量 $z$，压缩函数定义为：

$$
\operatorname{squash}(z)=
\frac{\|z\|^2}{1+\|z\|^2}\frac{z}{\|z\|}
$$

当 $z=0$ 时取零向量。为了避免除零，代码实现会在范数分母中加入很小的 $\epsilon$。

压缩函数不是普通单位向量归一化。较小输入的输出长度接近 0，较大输入的输出长度逐渐接近 1，从而同时保留方向和一定的强度信息。

### 2.4 行为到兴趣动态路由

**（1）共享映射**

设最多生成 $K$ 个兴趣向量 $u_1,\ldots,u_K$。行为 $e_i$ 先通过共享矩阵 $S\in\mathbb R^{D\times D}$ 映射：

$$
\widehat e_i=Se_i
$$

行为 $i$ 与兴趣 $j$ 的路由分数为：

$$
b_{ij}=u_j^T\widehat e_i
$$

同一矩阵 $S$ 用于所有行为—兴趣对，既能处理变长历史，也让多个兴趣处于同一向量空间。若每个位置和兴趣都使用独立映射，参数会依赖最大历史长度，也会破坏空间的一致性。

**（2）路由权重与兴趣更新**

对每条行为，在兴趣维度归一化：

$$
w_{ij}=\frac{\exp(b_{ij})}{\sum_{k=1}^{K_u'}\exp(b_{ik})}
$$

因此每条行为在有效兴趣之间的路由权重之和为 1。第 $j$ 个兴趣聚合所有行为：

$$
z_j=\sum_{i=1}^{L}w_{ij}\widehat e_i
$$

$$
u_j=\operatorname{squash}(z_j)
$$

随后重新计算 $b_{ij}=u_j^T\widehat e_i$，重复若干轮。相似行为会逐渐路由到相同兴趣，从而形成软聚类。

MIND 的更新是用当前兴趣重新计算路由分数。它不同于某些胶囊网络实现中把一致性分数累加到旧路由分数上的写法，复现时需要明确选择的算法版本。

**（3）对称性与初始化**

由于映射矩阵在兴趣间共享，如果所有 $b_{ij}$ 初始值完全相同，各兴趣会得到相同权重和向量，之后仍无法分化。因此需要非对称初始化。

模型训练需要可复现，线上用户向量也不能因每次请求重新采样而变化。下文实现会在模块创建时采样一次小幅高斯路由先验并保存为缓冲区；每次前向计算复用同一先验，而不是在请求过程中反复生成随机数。

**（4）动态兴趣数量**

行为较少的用户不需要占用全部 $K$ 个兴趣。MIND 使用启发式规则：

$$
K_u'=\max\left(1,\min\left(K,\left\lfloor\log_2L\right\rfloor\right)\right)
$$

代码将离散兴趣数明确取下整。空历史没有可路由行为，输出全部零向量和空兴趣掩码；此时由用户画像或冷启动策略承担召回。

该规则只依赖历史长度，不保证兴趣数量与真实语义簇数量一致。$K$ 是容量上限，需要通过召回效果、重复率、索引查询成本和用户活跃度共同调整。

### 2.5 用户表示网络

动态路由输出兴趣胶囊后，将同一个用户画像 $p_u$ 分别与每个兴趣拼接：

$$
v_{u,j}=f_{\mathrm{user}}([u_j;p_u])
$$

其中 $f_{\mathrm{user}}$ 是共享的多层感知机，输出维度与物品向量相同。对输出做 L2 归一化后，点积等价于余弦相似度，便于使用近似最近邻索引。

不同兴趣必须共享用户表示网络，否则各兴趣可能进入不同坐标空间，无法与同一个物品索引直接比较。无效兴趣在投影后重新置零，防止网络偏置把它变成非零召回向量。

### 2.6 标签感知注意力

训练样本包含目标物品 $e_y$，但一个用户有多个兴趣。标签感知注意力根据目标物品选择与它相关的兴趣：

$$
c_j=v_{u,j}^Te_y
$$

$$
\alpha_j=\frac{\exp(c_j^p)}{\sum_{k=1}^{K_u'}\exp(c_k^p)}
$$

$$
v_u(e_y)=\sum_{j=1}^{K_u'}\alpha_jv_{u,j}
$$

$p$ 控制分布尖锐程度。当采用有限 $p$ 时，需要注意负相似度的实数幂可能没有定义；工程实现通常约束 $p$ 为正整数，或改用带温度的 Softmax。下文采用 $p=1$ 的稳定形式。

硬标签感知注意力直接选择分数最大的兴趣：

$$
j^*=\arg\max_j c_j,\qquad v_u(e_y)=v_{u,j^*}
$$

硬选择收敛较快，但只有被选中的兴趣从当前样本获得标签路径梯度；软选择让多个兴趣参与训练。两种方式都只用于训练阶段将目标标签与多兴趣表示对齐。

目标物品不能作为线上用户塔输入。若服务时仍依赖标签感知层，就需要遍历物品库才能决定用户向量，失去了双塔召回的意义。

### 2.7 训练目标

完整物品 Softmax 为：

$$
P(i\mid u)=\frac{\exp(v_u^Te_i)}{\sum_{j\in\mathcal I}\exp(v_u^Te_j)}
$$

$$
\mathcal L=-\sum_{(u,i)\in\mathcal D}\log P(i\mid u)
$$

物品库很大时，分母无法逐样本完整计算，因此训练采用采样 Softmax。设为每个正样本采样 $M$ 个负物品，示例构造 $M+1$ 分类 Logits：

$$
\ell=[v_u^Te_y,v_u^Te_1^-,\ldots,v_u^Te_M^-]
$$

并令第 0 类为正确类别。严格的采样 Softmax还需要根据采样概率修正 Logits；下文为了展示模型结构使用均匀负采样分类损失，并明确要求负样本不等于正样本或补零 ID。

### 2.8 召回服务

训练完成后，物品塔离线生成全库向量并建立近邻索引。在线用户塔输出 $K_u'$ 个兴趣向量，每个向量分别召回 Top-$N$：

$$
R_u=\bigcup_{j=1}^{K_u'}\operatorname{ANN}(v_{u,j},N)
$$

合并时需要对同一物品去重，可以取它在各兴趣下的最高相似度：

$$
s(u,i)=\max_{1\le j\le K_u'}v_{u,j}^Te_i
$$

各兴趣固定召回同样数量并不一定合理。热门兴趣可能产生大量重复物品，长尾兴趣可能需要更多配额，实际系统还要结合去重后的数量、兴趣置信度和下游精排容量分配预算。

### 2.9 与兴趣精排模型的区别

| 模型 | 主要阶段 | 用户表示 | 候选物品的作用 |
| --- | --- | --- | --- |
| DIN | 精排 | 每个候选对应一个兴趣向量 | 激活相关历史行为 |
| DIEN | 精排 | 每个候选对应一条兴趣演化结果 | 控制 AUGRU 更新 |
| DSIN | 精排 | 每个候选对应会话兴趣聚合 | 激活相关会话 |
| MIND | 召回 | 候选无关的多个兴趣向量 | 训练时对齐兴趣；服务时在向量索引中被检索 |

MIND 不以行为时间顺序为核心，也不直接取代精排模型。它解决的是如何用有限数量的用户向量覆盖多样兴趣，并保持大规模向量检索的可行性。

## 3. 模型实现

### 3.1 核心代码实现

下面给出可运行的 PyTorch 示例。输入历史已经补齐为 `[B,L]`，真实物品 ID 为 `1..num_items`，0 专用于补零。

**（1）压缩函数与动态路由**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, eps=1e-8):
    squared_norm = (x * x).sum(dim=-1, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


class BehaviorToInterestRouting(nn.Module):
    def __init__(self, dim, max_interests=4, max_behaviors=50,
                 iterations=3, initial_std=0.01):
        super().__init__()
        self.max_interests = max_interests
        self.max_behaviors = max_behaviors
        self.iterations = iterations
        self.mapping = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "initial_logits",
            torch.randn(max_behaviors, max_interests) * initial_std,
        )

    def forward(self, behaviors, behavior_mask):
        batch_size, length, dim = behaviors.shape
        if length > self.max_behaviors:
            raise ValueError("历史长度超过 max_behaviors")
        lengths = behavior_mask.sum(dim=1)
        interest_counts = torch.zeros_like(lengths)
        nonempty = lengths > 0
        interest_counts[nonempty] = torch.clamp(
            torch.floor(torch.log2(lengths[nonempty].float())).long(),
            min=1, max=self.max_interests,
        )
        interest_mask = (
            torch.arange(self.max_interests, device=behaviors.device)[None, :]
            < interest_counts[:, None]
        )
        transformed = self.mapping(behaviors)
        logits = self.initial_logits[:length][None, :, :].expand(
            batch_size, -1, -1
        ).clone()
        interests = behaviors.new_zeros(batch_size, self.max_interests, dim)

        for _ in range(self.iterations):
            masked_logits = logits.masked_fill(
                ~interest_mask[:, None, :], -torch.inf
            )
            weights = torch.zeros_like(masked_logits)
            if nonempty.any():
                weights[nonempty] = torch.softmax(
                    masked_logits[nonempty], dim=-1
                )
            weights = weights * behavior_mask.unsqueeze(-1)
            candidates = torch.einsum("blk,bld->bkd", weights, transformed)
            interests = squash(candidates) * interest_mask.unsqueeze(-1)
            logits = torch.einsum("bld,bkd->blk", transformed, interests)
        return interests, interest_mask
```

空历史样本不会执行有效 Softmax，避免全为负无穷时产生 `NaN`。保存的初始化先验用于打破兴趣对称性，不参与优化，但会随模型检查点一起保存。

**（2）用户表示与标签感知选择**

```python
class LabelAwareAttention(nn.Module):
    def __init__(self, power=1, hard=False):
        super().__init__()
        if not isinstance(power, int) or power < 1:
            raise ValueError("power 必须是正整数")
        self.power, self.hard = power, hard

    def forward(self, interests, target, interest_mask):
        scores = torch.einsum("bkd,bd->bk", interests, target)
        scores = scores.pow(self.power).masked_fill(~interest_mask, -torch.inf)
        active = interest_mask.any(dim=1)
        output = interests.new_zeros(target.shape)
        if self.hard:
            chosen = scores[active].argmax(dim=1)
            output[active] = interests[active, chosen]
        else:
            weights = torch.zeros_like(scores)
            weights[active] = torch.softmax(scores[active], dim=1)
            output = torch.einsum("bk,bkd->bd", weights, interests)
        return output


class UserRepresentation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, interests, profile, interest_mask):
        expanded_profile = profile[:, None, :].expand_as(interests)
        output = self.network(torch.cat([interests, expanded_profile], dim=-1))
        output = F.normalize(output, dim=-1)
        return output * interest_mask.unsqueeze(-1)
```

**（3）完整模型**

```python
class MIND(nn.Module):
    def __init__(self, num_users, num_items, dim=16,
                 max_interests=4, max_behaviors=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(
            num_items + 1, dim, padding_idx=0
        )
        self.routing = BehaviorToInterestRouting(
            dim, max_interests, max_behaviors
        )
        self.user_network = UserRepresentation(dim)
        self.label_attention = LabelAwareAttention(power=1, hard=False)

    def encode_items(self, item_ids):
        return F.normalize(self.item_embedding(item_ids), dim=-1)

    def encode_user(self, users, history, lengths):
        if history.ndim != 2 or lengths.shape != (history.shape[0],):
            raise ValueError("history 应为 [B, L]，lengths 应为 [B]")
        if torch.any(lengths < 0) or torch.any(lengths > history.shape[1]):
            raise ValueError("历史长度超出范围")
        mask = (
            torch.arange(history.shape[1], device=history.device)[None, :]
            < lengths[:, None]
        )
        behaviors = self.item_embedding(history.masked_fill(~mask, 0))
        capsules, interest_mask = self.routing(behaviors, mask)
        profile = self.user_embedding(users)
        interests = self.user_network(capsules, profile, interest_mask)
        return interests, interest_mask

    def forward(self, users, history, lengths, targets, negatives):
        interests, interest_mask = self.encode_user(users, history, lengths)
        target_vectors = self.encode_items(targets)
        user_vector = self.label_attention(
            interests, target_vectors, interest_mask
        )
        negative_vectors = self.encode_items(negatives)
        positive_logits = (user_vector * target_vectors).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum(
            "bd,bmd->bm", user_vector, negative_vectors
        )
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, labels
```

`encode_user` 是服务阶段使用的用户塔接口，不需要目标物品。`forward` 中的标签感知注意力只负责训练时从多个兴趣中构造与正样本匹配的用户向量。

**（4）前向预测与训练示例**

```python
torch.manual_seed(42)
model = MIND(
    num_users=10, num_items=100, dim=8,
    max_interests=4, max_behaviors=6,
)
users = torch.tensor([0, 1, 2, 3])
history = torch.tensor([
    [1, 2, 3, 4, 5, 0],
    [10, 11, 12, 0, 0, 0],
    [20, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])
lengths = torch.tensor([5, 3, 1, 0])
targets = torch.tensor([8, 15, 30, 9])
negatives = torch.tensor([
    [41, 42, 43], [51, 52, 53],
    [61, 62, 63], [71, 72, 73],
])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
logits, labels = model(users, history, lengths, targets, negatives)
loss = nn.CrossEntropyLoss()(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
    user_interests, interest_mask = model.encode_user(
        users, history, lengths
    )
print("Interest shape:", user_interests.shape)
print("Valid interests:", interest_mask.sum(dim=1).tolist())
print("Training loss:", loss.item())
```

空历史用户没有有效兴趣，因此训练数据通常需要冷启动分支或至少一条历史。示例保留空历史样本用于验证数值边界，其用户向量为零，相应样本不会向用户塔提供有效兴趣梯度。

### 3.2 实验与工程注意事项

- **同时评估准确率与多样性**：除 Recall、HitRate 外，记录兴趣间相似度、跨兴趣召回重复率、类目覆盖率和最终去重候选数。
- **明确路由随机性**：初始化先验应随检查点保存，线上推理必须确定；不同初始化仍可能影响兴趣的排列与收敛结果。
- **审计动态兴趣数量**：按历史长度和活跃度分桶评估，避免短历史生成过多噪声兴趣，也避免高活跃用户被过小的 $K$ 限制。
- **校验负采样**：负物品不能等于正样本或补零 ID；非均匀采样需要考虑采样概率修正。
- **训练与服务严格解耦**：标签感知层不能进入线上用户塔。物品塔、用户塔和向量索引必须使用同一模型版本。
- **制定合并策略**：多个兴趣召回结果需要去重、分配配额和统一打分，再交给后续粗排或精排。

## 4. 总结

MIND 用多个向量表达用户的多样兴趣，并通过行为到兴趣动态路由对历史行为进行软聚类。共享映射保证兴趣处于同一空间，非对称初始化防止多个兴趣退化为相同向量，动态兴趣数量减少短历史用户的无效计算。

训练时，标签感知注意力把目标物品与相关兴趣对齐；服务时，目标物品不进入用户塔，多个兴趣向量分别查询物品索引。理解这一区别，是正确实现多兴趣召回和避免训练—服务不一致的关键。

## 5. 参考文献

[1] Li C, Liu Z, Wu M, et al. **Multi-Interest Network with Dynamic Routing for Recommendation at Tmall**. CIKM, 2019.

[2] Sabour S, Frosst N, Hinton G E. **Dynamic Routing Between Capsules**. NeurIPS, 2017.

[3] Covington P, Adams J, Sargin E. **Deep Neural Networks for YouTube Recommendations**. RecSys, 2016.

[4] Vaswani A, Shazeer N, Parmar N, et al. **Attention Is All You Need**. NeurIPS, 2017.
