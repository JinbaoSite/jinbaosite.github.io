# 多层感知机：从线性模型到非线性神经网络

多层感知机通过全连接层与非线性激活的交替组合，学习输入特征到目标之间的复杂映射。隐藏层将原始特征变换为可学习表示，输出层再根据回归、二分类或多分类任务生成预测。

[线性回归](/dl/linear_regression/)和[逻辑回归](/dl/logistic_regression/)提供了加权求和、损失函数与梯度优化的基础。多层感知机在此基础上增加非线性隐藏层，使特征变换也能随任务目标共同训练。

## 1. 背景

线性二分类模型的决策边界是超平面。如果类别关系依赖复杂交互，仅在原始特征上做线性加权可能无法分开样本。

异或任务是一个简单例子：

| 输入 $x_1$ | 输入 $x_2$ | 标签 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

两个正类位于正方形的一条对角线上，两个负类位于另一条对角线上，无法用一条直线严格分开。增加非线性隐藏层后，模型可以先构造不同区域的特征，再组合出分类结果。

隐藏层的价值在于改变表示空间。并非所有任务都需要深网络，线性模型、树模型和神经网络仍应在相同数据划分下比较。

## 2. 模型原理

### 2.1 单个神经元与全连接层

单个神经元先计算仿射变换，再执行激活函数：

$$
z=\mathbf w^T\mathbf x+b,\qquad h=\phi(z)
$$

一层包含多个神经元时，采用列向量约定：

$$
\mathbf z^{(l)}=W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)},\qquad
\mathbf h^{(l)}=\phi_l(\mathbf z^{(l)})
$$

若前一层宽度为 $d_{l-1}$，当前层宽度为 $d_l$，则 $W^{(l)}\in\mathbb R^{d_l\times d_{l-1}}$，偏置为 $d_l$ 维，参数量为：

$$
P_l=d_ld_{l-1}+d_l
$$

全连接表示当前层每个神经元都接收上一层所有分量。参数在不同样本之间共享，而不是为每条训练数据单独创建。

### 2.2 为什么需要非线性激活

若两层之间没有非线性：

$$
\mathbf h=W_1\mathbf x+\mathbf b_1
$$

$$
\mathbf z=W_2\mathbf h+\mathbf b_2
=(W_2W_1)\mathbf x+(W_2\mathbf b_1+\mathbf b_2)
$$

组合结果仍是一个仿射函数。继续堆叠也不会产生新的非线性表达能力，窄隐藏层还可能限制有效矩阵的秩。

加入非线性后，不同层可以逐步改变样本表示。常见激活包括：

| 激活 | 定义或性质 | 使用要点 |
| --- | --- | --- |
| 修正线性单元 | $\max(0,z)$ | 正半轴梯度稳定，负半轴可能长期不激活 |
| 双曲正切 | $\tanh(z)$ | 输出关于零对称，大幅输入可能饱和 |
| 逻辑函数 | $1/(1+e^{-z})$ | 常用于概率输出，隐藏层饱和会削弱梯度 |
| 高斯误差线性单元 | $z\Phi(z)$ | 平滑门控，常见于前馈模块 |

带修正线性激活的多层感知机是分段仿射函数。输出概率变换后，概率函数还会进一步变为非线性。

### 2.3 异或的显式构造

对二元输入，可以构造两个隐藏神经元：

$$
h_1=\max(0,x_1-x_2),\qquad
h_2=\max(0,x_2-x_1)
$$

则：

$$
f(\mathbf x)=h_1+h_2=|x_1-x_2|
$$

在四个异或输入上，$f$ 恰好等于标签。使用得分 $z=c(f-1/2)$、$c>0$，再以零为阈值，就得到正确类别。

这是证明小型非线性网络可以表达异或的显式构造，不意味着梯度训练一定找到同一组参数，也不意味着有限得分经过逻辑函数后会恰好输出 0 或 1。

### 2.4 多层网络与输出任务

设有 $L-1$ 个隐藏层，输出层为：

$$
\mathbf z=W^{(L)}\mathbf h^{(L-1)}+\mathbf b^{(L)}
$$

输出形式与损失应由任务共同确定：

| 任务 | 输出维度与含义 | 常见训练损失 |
| --- | --- | --- |
| 单目标回归 | 一个连续预测值 | 均方误差或绝对误差 |
| 二分类 | 一个未归一化得分 | 带得分输入的二元交叉熵 |
| 互斥多分类 | 每类一个得分 | 多类别交叉熵 |
| 多标签分类 | 每个标签一个独立得分 | 逐标签二元交叉熵 |

二分类训练可以直接把原始得分传入 `BCEWithLogitsLoss`，推理时再用逻辑函数得到概率。互斥多分类的 `CrossEntropyLoss` 同样接收原始得分，不要求提前执行概率归一化。

隐藏层激活和输出层变换承担不同角色。回归任务若对输出强行加修正线性激活，就会限制模型只能预测非负值，是否合理取决于目标定义。

### 2.5 反向传播

以单隐藏层、双曲正切激活和二分类为例：

$$
\mathbf h_i=\tanh(W_1\mathbf x_i+\mathbf b_1),\qquad
z_i=\mathbf w_2^T\mathbf h_i+b_2,\qquad p_i=\sigma(z_i)
$$

均值交叉熵对输出得分的导数为：

$$
\delta_i^{(2)}=\frac{p_i-y_i}{n}
$$

输出层梯度为：

$$
\nabla_{\mathbf w_2}J=\sum_i\delta_i^{(2)}\mathbf h_i,\qquad
\frac{\partial J}{\partial b_2}=\sum_i\delta_i^{(2)}
$$

隐藏层误差信号为：

$$
\boldsymbol\delta_i^{(1)}=
\left(\mathbf w_2\delta_i^{(2)}\right)\odot(\mathbf1-\mathbf h_i^2)
$$

因此：

$$
\nabla_{W_1}J=\sum_i\boldsymbol\delta_i^{(1)}\mathbf x_i^T,\qquad
\nabla_{\mathbf b_1}J=\sum_i\boldsymbol\delta_i^{(1)}
$$

每层将上游梯度乘以局部导数，便可按链式法则传播到所有参数。反向传播计算梯度，优化器使用梯度更新参数，两者不是同一操作。

与线性回归和逻辑回归的凸目标不同，含隐藏层的网络训练通常是非凸优化。表达某个函数的能力，不保证在有限数据和给定优化配置下能够学到它。

### 2.6 宽度、深度与表达能力

增加宽度可提供更多隐藏特征，增加深度则允许复用和组合中间表示。在适当激活及条件下，足够宽的单隐藏层网络能够在紧集上逼近连续函数，但这一结论不保证所需宽度、样本量、训练成本或泛化效果。

深度与宽度都增加容量，过大的网络可能过拟合，过小的网络可能欠拟合。参数规模应与数据量、噪声和任务结构一起选择，而不是单纯追求层数。

### 2.7 初始化、预处理与正则化

隐藏单元若使用完全相同的权重和偏置初始化，可能在训练中保持对称，无法学习不同特征。随机初始化打破对称性；合理方差则有助于控制逐层传播的数值尺度。

数值特征常用训练集均值与标准差标准化。类别特征需要独热或嵌入表示，缺失值处理也必须在训练与服务一致。验证、测试数据不能参与这些预处理参数的拟合。

常见容量控制方式包括权重惩罚、随机失活、限制隐藏层规模与验证集早停。随机失活在训练和评价模式下行为不同，评估前需要调用 `model.eval()`；关闭梯度记录则需要 `torch.no_grad()`，两者作用不同。

## 3. 模型实现

### 3.1 数据与可配置网络

PyTorch 示例生成四个簇组成的带扰动异或数据。标签由簇中心定义，训练、验证、测试各包含不同样本。数值预处理只使用训练集统计。

```python
import copy
import torch
from torch import nn


torch.manual_seed(42)
centers = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
center_labels = torch.tensor([0., 1., 1., 0.])
cluster = torch.arange(800) % 4
X = centers[cluster] + 0.12 * torch.randn(800, 2)
y = center_labels[cluster]
order = torch.randperm(len(X))
train_ids, valid_ids, test_ids = order[:480], order[480:640], order[640:]

mean = X[train_ids].mean(dim=0)
std = X[train_ids].std(dim=0, correction=0)
scale = torch.where(std > 0, std, torch.ones_like(std))
X_train = (X[train_ids] - mean) / scale
X_valid = (X[valid_ids] - mean) / scale
X_test = (X[test_ids] - mean) / scale
y_train, y_valid, y_test = y[train_ids], y[valid_ids], y[test_ids]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(16, 16), output_dim=1):
        super().__init__()
        if min(input_dim, output_dim, *hidden_dims) < 1:
            raise ValueError("layer widths must be positive")
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


model = MLP(input_dim=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

批量输入采用 `[批大小, 特征数]`，全连接层在最后一维执行变换。代码与公式等价，但输入样本按行排列，批量矩阵乘法对应 $XW^T+\mathbf b$。

网络 `[2,16,16,1]` 的参数量为：

$$
(2\times16+16)+(16\times16+16)+(16\times1+1)=337
$$

### 3.2 训练与验证集早停

```python
best_loss = float("inf")
best_state = None
patience = 60
waiting = 0
best_epoch = 0

for epoch in range(600):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train).squeeze(-1)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = criterion(model(X_valid).squeeze(-1), y_valid).item()

    if valid_loss < best_loss - 1e-5:
        best_loss = valid_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        waiting = 0
    else:
        waiting += 1
        if waiting >= patience:
            break

model.load_state_dict(best_state)
```

早停保存独立的参数副本，训练结束后恢复最佳验证损失对应的权重。仅保留 `state_dict()` 的浅引用，可能让所谓最佳权重随后续训练继续变化。

示例使用全批量更新以便复现。大数据可以改用小批量数据加载器，但需同时考虑批大小、数据打乱与训练评价模式。

### 3.3 测试与异或预测

```python
model.eval()
with torch.no_grad():
    test_logits = model(X_test).squeeze(-1)
    test_loss = criterion(test_logits, y_test).item()
    test_prediction = (test_logits >= 0).to(y_test.dtype)
    accuracy = test_prediction.eq(y_test).float().mean().item()

    center_probability = torch.sigmoid(
        model((centers - mean) / scale).squeeze(-1)
    )

print("best_epoch:", best_epoch)
print("parameter_count:", sum(parameter.numel() for parameter in model.parameters()))
print("test_loss:", round(test_loss, 6))
print("test_accuracy:", round(accuracy, 4))
print("xor_prediction:", (center_probability >= 0.5).int().tolist())
```

四个中心的类别预测应为 `[0,1,1,0]`，参数量为 337。合成任务用于检查非线性拟合和训练流程，不代表真实数据上的固定效果保证。测试集只在选定模型后评价，不能据此反复修改隐藏层或学习率。

### 3.4 训练诊断与计算成本

若训练损失不下降，优先检查标签、张量形状、梯度是否清空、参数是否交给优化器，以及输出与损失是否匹配。二分类预测 `[B,1]` 与标签 `[B]` 应显式对齐，避免形状误用。

训练效果好而验证效果差时，需要检查容量、数据划分和过拟合；训练验证都较差时，则可能是特征不足、模型欠拟合或优化未收敛。概率分数还应通过校准指标单独检查。

对批大小 $B$ 和各层宽度 $d_l$，前向矩阵乘法成本约为：

$$
O\left(B\sum_{l=1}^{L}d_{l-1}d_l\right)
$$

反向传播与之同阶，并需要保存中间激活；参数、梯度和优化器状态也占用内存。宽层尤其容易增加参数量，部署时需要结合延迟、吞吐量和内存选择网络规模。

## 4. 总结

多层感知机通过非线性隐藏层把固定线性表示扩展为可学习的特征变换。仿射层负责混合特征，激活函数引入非线性，损失函数定义任务目标，反向传播与优化器共同完成参数学习。

可靠训练需要统一输入预处理、输出语义和损失形状，并使用验证集选择容量与最佳权重。表达能力、优化效果与泛化性能需要分别检验，异或示例只提供最基础的可运行验证。

## 5. 参考文献

[1] Rumelhart D E, Hinton G E, Williams R J. Learning Representations by Back-propagating Errors. Nature, 1986, 323: 533–536.

[2] Hornik K, Stinchcombe M, White H. Multilayer Feedforward Networks Are Universal Approximators. Neural Networks, 1989, 2(5): 359–366.

[3] Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press, 2016.
