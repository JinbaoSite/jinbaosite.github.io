# DNN：深度神经网络的模型结构与实现

DNN 是深度神经网络的统称，可以涵盖多种具有多层可学习变换的结构。表格预测与推荐排序中，DNN 常指以全连接层为主体的深层前馈网络：输入编码形成统一向量，多层隐藏网络提取表示，任务头生成连续值或分类得分。

深层全连接 DNN 与[多层感知机](/dl/mlp/)在结构上没有明确分界。多层感知机强调基本非线性映射，DNN 则常用于描述更深的网络及其完整输入、主干和输出配置。层数达到某个固定值，并不是统一的命名标准。

## 1. 背景

实际预测模型的输入可能同时包含年龄、时长等数值字段，以及地区、品类等类别字段。直接把所有原始值相加无法表达字段含义，先统一编码再构建隐藏网络，才能形成可学习的端到端映射。

全连接 DNN 不预设图像的空间邻接或序列的时间关系，适合处理已经整理好的固定长度表示。若直接展平图像或长序列，参数和计算可能快速增长，也可能忽略有用结构，需要结合卷积、序列编码或注意力等模块设计输入。

## 2. 模型结构

### 2.1 整体架构

![DNN架构图](/img/dnn-architecture.svg)

> DNN架构图

图中展示输入层、单个隐藏层和输出层的三层全连接结构，用于说明层间连接；实际实现采用三个隐藏层，层宽与参数量在后续结构表中列出。

典型结构由四部分组成：

1. **输入处理**：缺失值处理、数值标准化和类别映射。
2. **特征编码**：类别嵌入与数值特征拼接，形成固定宽度输入。
3. **隐藏主干**：多层全连接与非线性激活，可选归一化和随机失活。
4. **任务输出**：根据目标配置输出维度、概率变换和损失函数。

输入处理中的固定统计量来自训练集，隐藏参数通过反向传播学习。训练与服务必须共享字段顺序、词表和预处理规则。

### 2.2 数值与类别输入

设有 $d_n$ 个数值特征，标准化后的向量为 $\mathbf x_n$。对类别字段 $j$，词表大小为 $V_j$，嵌入宽度为 $e_j$，嵌入表为：

$$
E_j\in\mathbb R^{(V_j+1)\times e_j}
$$

额外一行可用于未知类别。类别 ID $c_j$ 查表得到 $\mathbf e_j=E_j[c_j]$，再拼接为：

$$
\mathbf h^{(0)}=[\mathbf x_n;\mathbf e_1;\ldots;\mathbf e_q],\qquad
 d_0=d_n+\sum_{j=1}^{q}e_j
$$

类别编号只负责索引，不应直接解释为大小或距离。不同字段通常使用独立嵌入表，除非业务明确需要共享语义空间。

多值字段还需通过带掩码的求和、均值或注意力聚合成固定长度向量。均值池化必须处理空集合，补零项不能参与有效计数。

### 2.3 全连接隐藏层

第 $l$ 层的仿射变换为：

$$
\mathbf z^{(l)}=W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}
$$

其中 $W^{(l)}\in\mathbb R^{d_l\times d_{l-1}}$。加入激活后：

$$
\mathbf h^{(l)}=\phi_l(\mathbf z^{(l)})
$$

不同层学习对目标有用的中间表示，但不能不经分析就把每层强行解释成某种人工语义。非线性组合能够隐式表达交互，却不保证每个特征组合都能从有限数据中学好。

隐藏宽度可以逐层减小，也可以保持不变或先扩张后压缩。逐层减半不是必需规则；过窄瓶颈会限制信息容量，过宽网络则增加参数、过拟合和服务成本。

### 2.4 归一化、激活与随机失活

一个可选隐藏块为：

$$
\mathbf h^{(l)}=
\operatorname{Dropout}\left(\phi_l\left(\operatorname{LayerNorm}
(W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)})\right)\right)
$$

这只是一个明确的配置，不是所有 DNN 的标准顺序。归一化选择可结合[归一化层](/dl/normalization/)讨论，激活可结合[激活函数](/dl/activation_functions/)选择。

随机失活在训练时以概率 $p$ 将元素置零，保留元素缩放为 $1/(1-p)$ 倍，评价时恢复恒等操作。它不减少模型参数量，也不意味着部署时还随机删除神经元。

更深网络可以引入残差连接 $\mathbf h_{l+1}=\mathbf h_l+F(\mathbf h_l)$。输入输出宽度不同时，需要匹配形状的投影；仅增加跳连并不保证任意深度都易于训练。

### 2.5 输出头与损失

设最终隐藏表示为 $\mathbf h$，输出得分为 $\mathbf z=W_o\mathbf h+\mathbf b_o$：

| 任务 | 得分形状 | 推理变换 | 训练损失 |
| --- | --- | --- | --- |
| 单目标回归 | $[B,1]$ | 通常直接使用 | 均方误差或绝对误差 |
| 二分类 | $[B,1]$ | 逻辑函数 | 带得分的二元交叉熵 |
| 互斥多分类 | $[B,C]$ | 类别维归一化指数函数 | 多类别交叉熵 |
| 多标签分类 | $[B,K]$ | 每标签独立逻辑函数 | 逐标签二元交叉熵 |

回归输出是否需要非负约束由目标决定。二分类训练直接使用原始得分，不能先执行逻辑函数再传给 `BCEWithLogitsLoss`。

多任务网络可以共享主干，再分别连接任务头，目标为：

$$
\mathcal L=\sum_k\lambda_k\mathcal L_k
$$

不同任务标签缺失时需要独立掩码，任务间也可能梯度冲突。增加多个输出头并不自动解决目标尺度和样本空间差异。

### 2.6 逐层形状与参数量

以 6 个数值特征、两个类别字段和三个隐藏层为例：

| 模块 | 输出形状 | 参数量 |
| --- | --- | --- |
| 类别一嵌入：20 个有效 ID，宽度 8 | $[B,8]$ | $21\times8=168$ |
| 类别二嵌入：10 个有效 ID，宽度 4 | $[B,4]$ | $11\times4=44$ |
| 拼接 6 个数值特征 | $[B,18]$ | 0 |
| 全连接 $18\to64$ | $[B,64]$ | 1216 |
| 全连接 $64\to32$ | $[B,32]$ | 2080 |
| 全连接 $32\to16$ | $[B,16]$ | 528 |
| 三个层归一化 | 宽度不变 | $2(64+32+16)=224$ |
| 输出 $16\to1$ | $[B,1]$ | 17 |

总参数量为 4277，激活与随机失活没有可学习参数。未知类别行即使固定为零，也仍占据嵌入表存储。

一般全连接层参数量为 $d_{l-1}d_l+d_l$。批大小为 $B$ 时，主干前向乘法成本约为：

$$
O\left(B\sum_l d_{l-1}d_l\right)
$$

训练还需要保存激活、梯度和优化器状态。类别词表很大时，嵌入表可能成为主要存储开销，不能仅统计隐藏层。

### 2.7 前向传播的逐层计算

前向传播从输入开始，按依赖顺序计算隐藏表示、输出得分和损失。批量表示采用样本按行排列的约定：

$$
H^{(0)}\in\mathbb R^{B\times d_0},\qquad
W^{(l)}\in\mathbb R^{d_l\times d_{l-1}}
$$

仿射变换为：

$$
Z^{(l)}=H^{(l-1)}(W^{(l)})^T+\mathbf b^{(l)}
$$

偏置按样本维广播，输出形状为 $[B,d_l]$。对于代码采用的隐藏块，计算顺序是：

$$
Z^{(l)}\longrightarrow A^{(l)}=\operatorname{LayerNorm}(Z^{(l)})
\longrightarrow R^{(l)}=\operatorname{ReLU}(A^{(l)})
\longrightarrow H^{(l)}=\operatorname{Dropout}(R^{(l)})
$$

对某个样本的 $d_l$ 个分量，层归一化先计算：

$$
\mu=\frac1{d_l}\sum_jZ_j,\qquad
v=\frac1{d_l}\sum_j(Z_j-\mu)^2,\qquad
\widehat Z_j=\frac{Z_j-\mu}{\sqrt{v+\epsilon}}
$$

再执行 $A_j=\gamma_j\widehat Z_j+\beta_j$。激活将负值置零，训练时随机失活使用本次采样的二元掩码 $M$：

$$
H=\frac{M}{1-p}\odot R,\qquad M_j\sim\operatorname{Bernoulli}(1-p)
$$

评价时随机失活为恒等变换。前向过程中需要保存或能够重算各层输入、归一化统计与掩码，反向不能重新采样一份不同的掩码。

经过三个隐藏块，最终输出与二分类概率为：

$$
\mathbf z=H^{(3)}\mathbf w_o+b_o,\qquad
\mathbf q=\sigma(\mathbf z)
$$

对 $B$ 个样本的均值交叉熵：

$$
\mathcal L=\frac1B\sum_{i=1}^{B}
\left[\log(1+\exp(z_i))-y_iz_i\right]
$$

该表达式用于推导，实际计算使用稳定的融合损失。执行前向并不改变参数；训练模式下某些模块可能更新运行状态，但参数更新仍由优化器完成。

### 2.8 反向传播的逐层计算

反向传播从损失开始，先计算输出得分梯度：

$$
\delta_i=\frac{\partial\mathcal L}{\partial z_i}
=\frac{q_i-y_i}{B}
$$

这里的 $1/B$ 来自损失求均值。输出头的梯度为：

$$
\frac{\partial\mathcal L}{\partial\mathbf w_o}=(H^{(3)})^T\boldsymbol\delta,
\qquad
\frac{\partial\mathcal L}{\partial b_o}=\sum_i\delta_i
$$

$$
\frac{\partial\mathcal L}{\partial H^{(3)}}=
\boldsymbol\delta\mathbf w_o^T
$$

对每个隐藏块，沿随机失活、激活、归一化和仿射变换的逆序求导。设已收到 $G_H=\partial\mathcal L/\partial H$，训练时先计算：

$$
G_R=G_H\odot\frac{M}{1-p},\qquad
G_A=G_R\odot\mathbb I(A>0)
$$

零点使用框架的 ReLU 导数约定。归一化的可学习参数梯度为：

$$
\frac{\partial\mathcal L}{\partial\gamma_j}
=\sum_i(G_A)_{ij}\widehat Z_{ij},\qquad
\frac{\partial\mathcal L}{\partial\beta_j}=\sum_i(G_A)_{ij}
$$

令 $U=G_A\odot\gamma$。对每个样本独立沿特征维求均值，输入梯度可整理为：

$$
G_Z=\frac1{\sqrt{v+\epsilon}}
\left[U-\operatorname{mean}(U)
-\widehat Z\odot\operatorname{mean}(U\odot\widehat Z)\right]
$$

该式包含均值和方差依赖输入产生的导数，不能只把层归一化当作一个固定缩放系数。统计均值沿特征维计算，仿射参数梯度则沿样本维累加。

接着对全连接层求导：

$$
\frac{\partial\mathcal L}{\partial W^{(l)}}
=(G_Z^{(l)})^TH^{(l-1)},\qquad
\frac{\partial\mathcal L}{\partial\mathbf b^{(l)}}
=\sum_i(G_Z^{(l)})_{i,:}
$$

$$
\frac{\partial\mathcal L}{\partial H^{(l-1)}}=G_Z^{(l)}W^{(l)}
$$

梯度形状分别为 $[d_l,d_{l-1}]$、$[d_l]$ 和 $[B,d_{l-1}]$。沿隐藏层从后向前重复这一过程，直到输入拼接位置。损失的均值系数已经在输出梯度中计入，不在每层再次除以批大小。

拼接操作把梯度按原字段切开。对于类别字段，同一个类别 ID 被多条样本使用时，其嵌入梯度累加为：

$$
\frac{\partial\mathcal L}{\partial E_j[k]}
=\sum_{i:c_{ij}=k}\frac{\partial\mathcal L}{\partial\mathbf e_{ij}}
$$

整数类别 ID 本身没有这种连续梯度；模型学习的是嵌入表对应行。固定补零行不参与梯度更新。

所有梯度计算完成后，优化器才统一更新参数。若使用最基础的梯度下降：

$$
W\leftarrow W-\eta\nabla_W\mathcal L,\qquad
\mathbf b\leftarrow\mathbf b-\eta\nabla_{\mathbf b}\mathcal L
$$

AdamW 还会使用矩状态与解耦衰减，不能把它的实际更新直接等同于这一基础公式。反向应使用同一次前向对应的参数和中间量，不能每算完一层就修改权重再继续求上游梯度。

### 2.9 带具体数值的完整算例

为逐项核对数值，使用两个输入、两个 ReLU 隐藏单元和一个二分类输出，暂不加入归一化与随机失活。该简化只用于手算，完整隐藏块的求导按前两节处理。

设输入和标签为：

$$
\mathbf x=[1,2]^T,\quad y=1,\quad
W_1=\begin{bmatrix}1&-1\\0.5&1\end{bmatrix},\quad
\mathbf b_1=[0,0]^T
$$

输出权重为 $\mathbf w_2=[0.4,-0.8]^T$，偏置为 $b_2=0.2$。前向依次得到：

$$
\mathbf z_1=W_1\mathbf x+\mathbf b_1=[-1,2.5]^T,
\qquad\mathbf h=[0,2.5]^T
$$

$$
z_2=\mathbf w_2^T\mathbf h+b_2=-1.8,\qquad
q=\sigma(-1.8)\approx0.141851
$$

$$
\mathcal L=-\log q\approx1.952978
$$

单样本无需再除以批大小，反向从 $\delta=q-y\approx-0.858149$ 开始：

$$
\nabla_{\mathbf w_2}\mathcal L=\delta\mathbf h
\approx[0,-2.145372]^T,\qquad
\nabla_{b_2}\mathcal L\approx-0.858149
$$

隐藏表示梯度与激活前梯度为：

$$
\nabla_{\mathbf h}\mathcal L=\delta\mathbf w_2
\approx[-0.343260,0.686519]^T
$$

$$
\boldsymbol\delta_1=
\nabla_{\mathbf h}\mathcal L\odot[0,1]^T
\approx[0,0.686519]^T
$$

因此：

$$
\nabla_{W_1}\mathcal L=\boldsymbol\delta_1\mathbf x^T
\approx\begin{bmatrix}0&0\\0.686519&1.373038\end{bmatrix},\qquad
\nabla_{\mathbf b_1}\mathcal L\approx[0,0.686519]^T
$$

第一隐藏单元在该样本上不激活，因此对应梯度为零，但它仍可能被其他样本激活。取学习率 $0.1$，输出权重更新为约 $[0.4,-0.585463]^T$，输出偏置更新为 $0.285815$；其他参数同样按其梯度同步更新。

## 3. 模型实现

### 3.1 输入编码与主干

实现使用 PyTorch，数值输入为 `[B,6]`，类别输入为整数 `[B,2]`。各类别字段的 0 号 ID 作为固定零向量回退；有效编号分别为 `1..20` 和 `1..10`。真实数据的未知值应在查表之前映射到 0。

```python
import copy
import torch
from torch import nn


class DNN(nn.Module):
    def __init__(self, num_numeric=6, cardinalities=(20, 10),
                 embedding_dims=(8, 4), hidden_dims=(64, 32, 16),
                 output_dim=1, dropout=0.1):
        super().__init__()
        if len(cardinalities) != len(embedding_dims):
            raise ValueError("one embedding dimension is required for each field")
        if min(num_numeric, output_dim, *cardinalities, *embedding_dims, *hidden_dims) < 1:
            raise ValueError("dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_numeric = num_numeric
        self.num_categories = len(cardinalities)
        self.embeddings = nn.ModuleList([
            nn.Embedding(size + 1, dim, padding_idx=0)
            for size, dim in zip(cardinalities, embedding_dims)
        ])
        width = num_numeric + sum(embedding_dims)
        blocks = []
        for next_width in hidden_dims:
            blocks.append(nn.Sequential(
                nn.Linear(width, next_width),
                nn.LayerNorm(next_width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            width = next_width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(width, output_dim)

    def forward(self, numeric, categorical):
        if numeric.ndim != 2 or numeric.shape[1] != self.num_numeric:
            raise ValueError("invalid numeric shape")
        if categorical.shape != (numeric.shape[0], self.num_categories):
            raise ValueError("invalid categorical shape")
        features = [numeric]
        for index, embedding in enumerate(self.embeddings):
            features.append(embedding(categorical[:, index]))
        hidden = torch.cat(features, dim=-1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)
```

数值输入应为有限浮点值，类别索引必须落入各自词表范围。模型保持输出为 `[B,output_dim]`，由任务代码决定是否压缩最后一维。

### 3.2 合成数据与训练

生成包含数值交互和类别作用的二分类任务，再划分训练、验证与测试集。标签按概率采样，因此存在不可完全消除的噪声。

```python
torch.manual_seed(42)
numeric = torch.randn(900, 6)
categorical = torch.stack([
    torch.randint(1, 21, (900,)),
    torch.randint(1, 11, (900,)),
], dim=1)
score = (1.5 * numeric[:, 0] - numeric[:, 1] +
         0.8 * numeric[:, 2] * numeric[:, 3] +
         0.7 * (categorical[:, 0] <= 10).float() -
         0.6 * (categorical[:, 1] <= 5).float())
labels = torch.bernoulli(torch.sigmoid(score))
order = torch.randperm(len(labels))
train_ids, valid_ids, test_ids = order[:540], order[540:720], order[720:]
mean = numeric[train_ids].mean(dim=0)
std = numeric[train_ids].std(dim=0, correction=0)
scale = torch.where(std > 0, std, torch.ones_like(std))
normalized = (numeric - mean) / scale

model = DNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()
best_loss, best_state, waiting = float("inf"), None, 0

for epoch in range(120):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(normalized[train_ids], categorical[train_ids]).squeeze(-1)
    loss = criterion(logits, labels[train_ids])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_logits = model(normalized[valid_ids], categorical[valid_ids]).squeeze(-1)
        valid_loss = criterion(valid_logits, labels[valid_ids]).item()
    if valid_loss < best_loss - 1e-5:
        best_loss = valid_loss
        best_state = copy.deepcopy(model.state_dict())
        waiting = 0
    else:
        waiting += 1
        if waiting >= 20:
            break

model.load_state_dict(best_state)
```

示例使用全批量训练和统一参数组。固定零嵌入初始化为零，梯度不写入该行，乘法式权重衰减也会保持零；它是一种回退选择，不等同于学习到未知类别的最佳表示。

### 3.3 测试与结构核对

```python
model.eval()
with torch.no_grad():
    test_logits = model(normalized[test_ids], categorical[test_ids]).squeeze(-1)
    test_loss = criterion(test_logits, labels[test_ids]).item()
    accuracy = ((test_logits >= 0).float() == labels[test_ids]).float().mean().item()

    hidden = torch.cat([
        normalized[test_ids],
        *[embedding(categorical[test_ids, index])
          for index, embedding in enumerate(model.embeddings)],
    ], dim=-1)
    shapes = [tuple(hidden.shape)]
    for block in model.blocks:
        hidden = block(hidden)
        shapes.append(tuple(hidden.shape))

print("hidden_shapes:", shapes)
print("output_shape:", tuple(model.head(hidden).shape))
print("parameter_count:", sum(parameter.numel() for parameter in model.parameters()))
print("test_loss_accuracy:", round(test_loss, 4), round(accuracy, 4))
```

隐藏形状依次为 `[180,18]`、`[180,64]`、`[180,32]`、`[180,16]`，输出为 `[180,1]`，参数量为 4277。测试指标只描述这份合成数据，不构成实际业务中的效果保证。

### 3.4 手推梯度与自动求导核对

```python
demo_x = torch.tensor([1., 2.], dtype=torch.float64)
demo_w1 = torch.tensor([[1., -1.], [0.5, 1.]], dtype=torch.float64,
                       requires_grad=True)
demo_b1 = torch.zeros(2, dtype=torch.float64, requires_grad=True)
demo_w2 = torch.tensor([0.4, -0.8], dtype=torch.float64, requires_grad=True)
demo_b2 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
demo_z1 = demo_w1 @ demo_x + demo_b1
demo_h = torch.relu(demo_z1)
demo_z2 = demo_w2 @ demo_h + demo_b2
demo_target = torch.tensor(1., dtype=torch.float64)
demo_loss = nn.functional.binary_cross_entropy_with_logits(demo_z2, demo_target)
demo_loss.backward()

with torch.no_grad():
    delta = torch.sigmoid(demo_z2) - demo_target
    hidden_delta = delta * demo_w2 * (demo_z1 > 0)
    assert torch.allclose(demo_w2.grad, delta * demo_h)
    assert torch.allclose(demo_b2.grad, delta)
    assert torch.allclose(demo_w1.grad, hidden_delta[:, None] * demo_x[None, :])
    assert torch.allclose(demo_b1.grad, hidden_delta)
print("manual_loss:", round(demo_loss.item(), 6))
print("manual_gradient_check:", True)
```

归一化与随机失活的局部梯度也可以独立核对。固定同一份掩码，不在反向时重新采样：

```python
torch.manual_seed(7)
block_z = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
block_gamma = torch.randn(4, dtype=torch.float64, requires_grad=True)
block_beta = torch.randn(4, dtype=torch.float64, requires_grad=True)
block_eps, drop_probability = 1e-5, 0.2
block_mask = (torch.rand(3, 4) >= drop_probability).double()
block_mean = block_z.mean(dim=-1, keepdim=True)
block_variance = block_z.var(dim=-1, correction=0, keepdim=True)
block_hat = (block_z - block_mean) / torch.sqrt(block_variance + block_eps)
block_a = block_hat * block_gamma + block_beta
block_h = torch.relu(block_a) * block_mask / (1 - drop_probability)
upstream = torch.randn_like(block_h)
(block_h * upstream).sum().backward()

with torch.no_grad():
    grad_a = upstream * block_mask / (1 - drop_probability) * (block_a > 0)
    grad_u = grad_a * block_gamma
    grad_z = (grad_u - grad_u.mean(dim=-1, keepdim=True) -
              block_hat * (grad_u * block_hat).mean(dim=-1, keepdim=True))
    grad_z = grad_z / torch.sqrt(block_variance + block_eps)
    assert torch.allclose(block_z.grad, grad_z, atol=1e-10)
    assert torch.allclose(block_gamma.grad, (grad_a * block_hat).sum(dim=0))
    assert torch.allclose(block_beta.grad, grad_a.sum(dim=0))
print("normalization_activation_dropout_gradient_check:", True)
```

两组检查分别覆盖完整手算网络和隐藏块内部的求导链。第二组将归一化公式显式展开，核对的数学运算与 `LayerNorm` 一致；训练模型继续使用框架模块。

### 3.5 结构设计与部署边界

1. **输入语义优先于网络深度**：未来信息泄漏、类别映射错误和缺失处理不一致无法通过增加隐藏层修复。
2. **容量需要独立验证**：改变宽度、层数、归一化或随机失活后，重新检查训练与验证曲线，避免只观察训练误差。
3. **区分任务得分与概率**：类别权重、负采样或训练分布改变后，概率校准需要单独验证。
4. **保存完整推理依赖**：模型之外还需保存字段顺序、词表、数值统计、输出标签映射和分类阈值。
5. **评估服务成本**：分别测量嵌入查表、主干矩阵运算与数据传输，按目标批大小测试延迟和吞吐量。

## 4. 总结

全连接 DNN 可以拆成输入处理、特征编码、隐藏主干和任务输出四部分。嵌入把类别字段映射到可学习向量，隐藏层通过非线性组合提取表示，输出头与损失共同定义预测任务。

结构可靠性来自形状、参数量与任务语义的一致性。深度只是容量配置的一部分，正确的数据划分、预处理和部署依赖同样决定模型是否可用。

## 5. 参考文献

[1] Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press, 2016.

[2] Bishop C M. Pattern Recognition and Machine Learning. Springer, 2006.

[3] Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 2014, 15: 1929–1958.
