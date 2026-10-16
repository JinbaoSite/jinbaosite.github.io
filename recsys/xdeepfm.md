# xDeepFM：显式与隐式特征交互融合

xDeepFM（eXtreme Deep Factorization Machine）发表于 KDD 2018，论文题为 *xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems* [1]。

xDeepFM 的核心是**压缩交互网络 CIN**：在保留字段嵌入向量结构的同时，逐层构造显式高阶交互，并与线性模型、深度网络联合学习，用于点击率预测等精排任务。

## 1. 背景

点击率预测常常依赖特征组合。例如，用户所在城市、商品类别和访问时段共同影响点击意愿。单独增加某个特征的权重，难以描述这种条件关系。

[DeepFM](/recsys/deepfm/) 使用 FM 建模显式二阶交互，并通过深度网络学习隐式高阶关系。深度分支能够表达复杂函数，但没有为每一层指定明确的交互阶数。[DCN](/recsys/dcn/) 则通过递推交叉层构造有界阶数的交互，不过计算建立在拼接向量的坐标上。

xDeepFM 引入 CIN，将一个字段的嵌入向量作为交互单元。它让上一层特征向量与输入字段向量做逐元素乘法，再把大量交互向量压缩为少量输出向量。堆叠多层后，模型可以在受控宽度下构造更高阶的组合。

## 2. 模型架构

### 2.1 整体网络结构

![xDeepFM架构图](/img/xdeepfm-architecture.svg)

> xDeepFM架构图

模型包含三个并行分支：

- **线性分支**：直接对输入特征加权，学习一阶贡献。
- **压缩交互网络**：显式构造向量级高阶交互，对各层输出池化后参与预测。
- **深度网络**：将字段嵌入展平，通过多层非线性变换学习隐式关系。

CIN 与深度网络共享输入嵌入，三路分数相加后经过 Sigmoid 得到点击概率。线性分支使用自己的标量权重，不是对嵌入向量简单求和。

### 2.2 输入层

设输入有 $m$ 个字段，各字段嵌入维度统一为 $D$，将它们按行堆叠：

$$
X^0=\begin{bmatrix}e_1^T\\e_2^T\\\vdots\\e_m^T\end{bmatrix}\in\mathbb R^{m\times D}
$$

单值类别字段通过查表得到 $e_i$。多值字段可以先池化为一个向量；数值字段可以用归一化后的数值缩放一个可学习向量。这里需要明确缺失值处理和池化规则，使训练与预测阶段保持一致。

**CIN 的各字段必须具有相同的嵌入维度**，才能沿对应坐标相乘。深度分支接收 $\operatorname{vec}(X^0)\in\mathbb R^{mD}$，CIN 分支则保留 $m\times D$ 的二维结构。

### 2.3 压缩交互网络

**（1）交互与压缩**

令第 $k$ 层输出为 $X^k\in\mathbb R^{H_k\times D}$，其中 $H_k$ 表示输出特征向量数量，且 $H_0=m$。第 $h$ 个输出向量为：

$$
X^k_{h,:}=\sum_{i=1}^{H_{k-1}}\sum_{j=1}^{m}W^k_{hij}\left(X^{k-1}_{i,:}\odot X^0_{j,:}\right)
$$

其中 $W^k\in\mathbb R^{H_k\times H_{k-1}\times m}$，$\odot$ 表示逐元素乘法。每层都与初始字段矩阵 $X^0$ 交互，而不是只让上一层内部的向量两两交互。

计算过程可以分为两步。首先构造交互张量：

$$
Z^k_{i,j,d}=X^{k-1}_{i,d}X^0_{j,d}
$$

然后对字段对所在的两个轴加权求和：

$$
X^k_{h,d}=\sum_{i,j}W^k_{hij}Z^k_{i,j,d}
$$

“压缩”指将 $H_{k-1}m$ 个交互向量映射为 $H_k$ 个向量。压缩的是向量数量，嵌入维度 $D$ 在层间保持不变。

**（2）向量级交互的含义**

同一个字段对的权重 $W^k_{hij}$ 在所有嵌入坐标 $d$ 上共享，因此它整体控制一对向量的贡献。这里并不执行两个嵌入向量之间的完整外积，也不会在 CIN 层内直接相乘不同坐标 $d\ne d'$。

把交互张量变形为 $[B,H_{k-1}m,D]$ 后，可以用卷积核大小为 1 的一维卷积实现压缩。此时 $H_{k-1}m$ 是输入通道数，$D$ 是长度轴；卷积仅混合通道，并不滑动组合相邻嵌入坐标。

**（3）交互阶数如何增加**

第一层把两个输入向量相乘，得到二阶交互。第二层再乘一个输入向量，例如：

$$
X^2_{h,:}=\sum_{i,j,a,b}W^2_{hij}W^1_{iab}\left(e_a\odot e_b\odot e_j\right)
$$

因此，在不加入偏置和非线性激活的基础 CIN 中，第 $k$ 层表示关于输入嵌入的 $k+1$ 阶齐次多项式；收集 $T$ 层输出，就能得到二阶到 $T+1$ 阶的交互。

这里的阶数针对嵌入坐标的乘积，不等同于包含同样数量的不同字段。求和允许重复字段，例如 $e_i\odot e_i$，所以不能把每个三阶项都理解为三个不同字段的组合。

此外，高阶系数由多层权重的乘积共同决定，并非所有高阶项都有独立参数。若在交叉层加入非线性激活，就不能继续直接套用有限次多项式的解释；加入偏置也会改变齐次性。下文实现采用无偏置、无激活的交叉层。

### 2.4 多层池化与输出

对每层的每个向量沿嵌入维度求和：

$$
p^k_h=\sum_{d=1}^{D}X^k_{h,d}
$$

然后拼接各层池化结果：

$$
p^+=[p^1;p^2;\cdots;p^T]\in\mathbb R^{\sum_{k=1}^{T}H_k},\qquad s_{\mathrm{cin}}=w_{\mathrm{cin}}^Tp^+
$$

池化发生在输出分支，传给下一层的仍是完整矩阵 $X^k$。如果先把上一层压成标量再递推，就会丢失后续逐坐标交互所需的信息。

收集所有层输出，使预测层能够同时使用不同阶数的信号。下面代码将每层全部向量用于继续交互及输出，不额外划分两组通道。

### 2.5 线性分支与深度分支

设原始编码特征为 $a$，线性分数为：

$$
s_{\mathrm{linear}}=w_{\mathrm{linear}}^Ta
$$

对于单值类别字段，可以为每个类别学习一个标量，再跨字段求和。它等价于在独热编码上应用线性模型，无需显式构造稀疏大向量。

深度网络从展平后的嵌入开始：

$$
h^0=\operatorname{vec}(X^0),\qquad h^{l+1}=\operatorname{ReLU}(A_lh^l+b_l)
$$

最终预测为：

$$
s=s_{\mathrm{linear}}+w_{\mathrm{cin}}^Tp^++w_{\mathrm{deep}}^Th^L+b,\qquad \hat y=\sigma(s)
$$

三路相加的是分数，而不是分别归一化后的概率。训练时直接使用分数 $s$ 与 `BCEWithLogitsLoss`，预测时再计算 Sigmoid。

### 2.6 与相关模型的区别

| 模型 | 显式交互部分 | 深度分支 | 主要特点 |
| --- | --- | --- | --- |
| FM | 字段嵌入内积构造二阶项 | 无 | 低成本建模二阶关系 |
| DeepFM | FM 二阶项 | 有 | 二阶显式交互与隐式交互共享嵌入 |
| DCN | 拼接向量上的递推交叉 | 有 | 逐层提高显式交互阶数 |
| xDeepFM | 字段向量逐元素乘积与压缩 | 有 | 跨嵌入坐标共享交互权重，汇集多阶输出 |

FM 的交互项是 $\sum_{i<j}\langle e_i,e_j\rangle$。单层 CIN 若只保留一个输出通道，并将权重设为选择 $i<j$ 的固定权重，再进行求和池化，就可以恢复该二阶项。但一般 CIN 还包含可学习权重和重复字段交互，不能把任意单层 CIN 都视为标准 FM。

### 2.7 参数量与计算成本

无偏置的第 $k$ 层有 $H_kH_{k-1}m$ 个参数。包含 CIN 的线性输出权重时，参数量为：

$$
P_{\mathrm{cin}}=\sum_{k=1}^{T}\left(H_kH_{k-1}m+H_k\right)
$$

这里不包含共享嵌入、深度分支、线性分支和全局偏置。CIN 交叉权重的数量不依赖 $D$，但计算量和中间张量大小依赖 $D$。

直接构造交互张量时，第 $k$ 层每批中间元素数为 $BH_{k-1}mD$，压缩计算量约为 $O(BDH_kH_{k-1}m)$。因此，“参数与嵌入维度无关”不意味着增加嵌入维度不会增加显存和延迟。

例如 $m=20,D=16,H_1=H_2=32$，两层交叉权重加 CIN 输出权重共 $12\,800+20\,480+64=33\,344$ 个参数。若 $B=256$，第二层交互张量含 $2\,621\,440$ 个元素，单个 float32 张量约占 10 MiB，训练还需要保存其他激活与梯度。

### 2.8 损失函数

点击率预测使用二元交叉熵：

$$
\mathcal L=-\frac{1}{N}\sum_{n=1}^{N}\left[y_n\log\hat y_n+(1-y_n)\log(1-\hat y_n)\right]
$$

还可以针对选定参数施加正则化。评估时同时观察 AUC 与 LogLoss，前者反映区分能力，后者衡量概率预测误差。下面示例只实现交叉熵训练，不把模拟数据上的损失作为模型效果结论。

## 3. 模型实现

### 3.1 核心代码实现

下面使用 PyTorch 实现基础 CIN 和完整 xDeepFM。为了清楚展示字段结构，输入限定为每个字段一个类别 ID；各字段 ID 从 0 开始，分别使用自己的嵌入表。

**（1）压缩交互网络**

```python
import torch
import torch.nn as nn


class CIN(nn.Module):
    def __init__(self, num_fields, layer_sizes=(32, 32)):
        super().__init__()
        if num_fields < 1 or not layer_sizes or any(h < 1 for h in layer_sizes):
            raise ValueError("字段数与各层宽度必须为正整数")
        self.layers = nn.ModuleList()
        previous = num_fields
        for width in layer_sizes:
            # 在所有嵌入坐标上共享相同的通道混合权重。
            self.layers.append(nn.Conv1d(
                previous * num_fields, width, kernel_size=1, bias=False
            ))
            previous = width
        self.output_dim = sum(layer_sizes)

    def forward(self, x0):
        # x0: [B, M, D]；hidden: [B, H, D]
        hidden = x0
        pooled = []
        for layer in self.layers:
            products = torch.einsum("bhd,bmd->bhmd", hidden, x0)
            products = products.flatten(1, 2)  # [B, H*M, D]
            hidden = layer(products)
            pooled.append(hidden.sum(dim=-1))
        return torch.cat(pooled, dim=1)
```

这里没有在交叉层中加入激活与偏置，与前面的阶数推导保持一致。`flatten(1, 2)` 将上一层通道和输入字段合并，不能把嵌入坐标也合并到通道中。

**（2）完整模型**

```python
class XDeepFM(nn.Module):
    def __init__(self, field_dims, embedding_dim=16,
                 cin_sizes=(32, 32), deep_sizes=(128, 64)):
        super().__init__()
        if not field_dims or any(n < 1 for n in field_dims):
            raise ValueError("每个字段的类别数量必须为正整数")
        if embedding_dim < 1 or not deep_sizes or any(h < 1 for h in deep_sizes):
            raise ValueError("嵌入维度与深度网络宽度必须为正整数")
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embedding_dim) for n in field_dims
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(n, 1) for n in field_dims
        ])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        for embedding in self.linear_embeddings:
            nn.init.zeros_(embedding.weight)

        self.cin = CIN(len(field_dims), cin_sizes)
        self.cin_head = nn.Linear(self.cin.output_dim, 1, bias=False)
        layers = []
        width = len(field_dims) * embedding_dim
        for next_width in deep_sizes:
            layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        self.deep = nn.Sequential(*layers)
        self.deep_head = nn.Linear(width, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[1] != len(self.embeddings):
            raise ValueError("输入形状必须为 [批量大小, 字段数]")
        x0 = torch.stack([
            embedding(ids[:, i])
            for i, embedding in enumerate(self.embeddings)
        ], dim=1)
        linear = torch.stack([
            embedding(ids[:, i])
            for i, embedding in enumerate(self.linear_embeddings)
        ], dim=1).sum(dim=1)
        cin_score = self.cin_head(self.cin(x0))
        deep_score = self.deep_head(self.deep(x0.flatten(1)))
        return (linear + cin_score + deep_score + self.bias).squeeze(-1)
```

三路输出都是 `[B, 1]`，最后仅移除最后一个维度，因此批量大小为 1 时仍返回 `[1]`。CIN 与深度网络使用同一份 `x0`，反向传播时共同更新字段嵌入。

**（3）前向预测与训练示例**

```python
torch.manual_seed(42)
field_dims = [100, 50, 20, 10]
model = XDeepFM(field_dims, embedding_dim=8,
                cin_sizes=(16, 16), deep_sizes=(32, 16))
ids = torch.stack([torch.randint(n, (16,)) for n in field_dims], dim=1)
labels = torch.randint(0, 2, (16,)).float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
logits = model(ids)
loss = nn.BCEWithLogitsLoss()(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(ids))
print("Prediction shape:", probabilities.shape)
print("Training loss:", loss.item())
```

### 3.2 实验与工程注意事项

- **固定输入再比较模型**：在相同数据划分、字段编码和训练预算下比较 DeepFM、纯 CIN 与 xDeepFM，并通过移除分支检验各部分收益。
- **分别调整深度和宽度**：CIN 深度影响交互阶数，宽度影响可保留的交互向量数量；两者都会改变计算成本，增加层数并不保证效果提升。
- **关注中间张量**：字段较多时，显式交互张量可能成为显存瓶颈。可以考虑分块收缩或低秩权重，但需要验证结果一致性与真实延迟。
- **检查数值尺度**：重复乘法会放大输入尺度差异，也可能让高阶信号衰减。应检查归一化、嵌入初始化、各分支输出和梯度分布。
- **防止数据泄漏**：根据业务时间划分训练与验证数据，避免将未来行为、全量统计或标签相关信息引入特征。

## 4. 总结

xDeepFM 通过 CIN 将显式特征交互从二阶扩展到更高阶。它保留字段向量结构，在对应嵌入坐标上相乘，再用跨坐标共享的权重压缩交互向量，最终汇集不同层的输出。

理解 CIN 的关键在于区分**字段轴、嵌入坐标轴和输出通道轴**：交互发生在字段向量之间，权重在嵌入坐标上共享，压缩控制输出通道数量。结合线性分支与深度网络后，模型同时学习一阶贡献、显式高阶交互和隐式非线性关系。

## 5. 参考文献

[1] Lian J, Zhou X, Zhang F, et al. **xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems**. KDD, 2018.

[2] Guo H, Tang R, Ye Y, et al. **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction**. IJCAI, 2017.

[3] Wang R, Fu B, Fu G, Wang M. **Deep & Cross Network for Ad Click Predictions**. AdKDD, 2017.

[4] Rendle S. **Factorization Machines**. ICDM, 2010.
